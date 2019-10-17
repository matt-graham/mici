"""Exception types."""


class IntegratorError(RuntimeError):
    """Error raised when integrator step fails."""


class NonReversibleStepError(IntegratorError):
    """Error raised when integrator step fails reversibility check."""


class ConvergenceError(IntegratorError):
    """Error raised when solver fails to converge within allowed iterations."""
