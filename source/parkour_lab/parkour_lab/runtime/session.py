"""Durable, ordered environment/application cleanup; no hardware management."""

import os
import sys

from .console import log_event


def exit_native_process(code):
    """Exit a standalone native run after explicit cleanup, not Python finalizers.

    Kit releases its framework in close(); later native destructors can then
    access unloaded plugins. Files and native resources must already be closed.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            try:
                stream.flush()
            except (OSError, ValueError):
                code = code or 2
    os._exit(code)


def finish_session(env, app, report, publish, code):
    """Close in order; preserve the last durable receipt if Kit exits in close()."""
    resources = (("environment", env), ("application", app))
    report["session_status"] = report["status"]
    if code == 0:
        report["status"] = "SESSION_COMPLETED_CLEANUP_PENDING"
    report["cleanup"] = {
        name: "pending" if resource is not None else "not_created"
        for name, resource in resources
    }
    log_event("cleanup_started", status=report["session_status"])

    def save():
        nonlocal code
        try:
            publish()
        except Exception as error:
            code = 2
            report.update(status="ERROR", report_write_error=str(error))
            log_event("report_write_failed", error=str(error))
            try:
                print(f"Could not publish run report: {error}", flush=True)
            except OSError:
                pass  # A broken logging pipe must not prevent resource cleanup.

    save()
    for name, resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
            report["cleanup"][name] = "complete"
        except Exception as error:
            report["cleanup"][name] = f"ERROR: {error}"
            code = 2
        log_event("resource_closed", resource=name, status=report["cleanup"][name])
        save()
    report["status"] = report["session_status"] if code == 0 else "ERROR"
    save()
    log_event(
        "session_finished",
        status=report["status"],
        returncode=code,
        cleanup=report["cleanup"],
        interactions=report.get("interactions", report.get("collected_interactions")),
    )
    return code
