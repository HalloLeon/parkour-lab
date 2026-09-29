"""Durable, ordered environment/application cleanup; no hardware management."""


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

    def save():
        nonlocal code
        try:
            publish()
        except Exception as error:
            code = 2
            report.update(status="ERROR", report_write_error=str(error))
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
        save()
    report["status"] = report["session_status"] if code == 0 else "ERROR"
    save()
    return code
