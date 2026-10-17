"""Frozen evaluation and versioned evidence interpretation, independent of learners."""

from .cases import EvaluationCase, case_bank

__all__ = ["EvaluationCase", "case_bank", "evaluate", "score_capture"]


def evaluate(policy, cases, output, **kwargs):
    """Run an already-loaded policy after the caller starts AppLauncher."""
    from .runner import evaluate as run

    return run(policy, cases, output, **kwargs)


def score_capture(folder):
    """Score saved traces without starting or importing the simulator."""
    from .capture import score_capture as score

    return score(folder)
