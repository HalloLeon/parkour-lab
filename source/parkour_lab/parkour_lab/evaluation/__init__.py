"""Frozen evaluation and versioned evidence interpretation, independent of learners."""

from .cases import EvaluationCase, case_bank
from .robustness import robustness_bank

__all__ = [
    "EvaluationCase",
    "case_bank",
    "robustness_bank",
    "evaluate",
    "score_capture",
    "report_campaign",
]


def evaluate(policy, cases, output, **kwargs):
    """Run an already-loaded policy after the caller starts AppLauncher."""
    from .runner import evaluate as run

    return run(policy, cases, output, **kwargs)


def score_capture(folder):
    """Score saved traces without starting or importing the simulator."""
    from .capture import score_capture as score

    return score(folder)


def report_campaign(path):
    """Check a declared campaign and summarize independent training seeds offline."""
    from .campaign import report_campaign as report

    return report(path)
