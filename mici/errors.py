"""Exception types."""


class Error(RuntimeError):
    """Base class for errors."""


class IntegratorError(Error):
    """Error raised when integrator step fails."""


class NonReversibleStepError(IntegratorError):
    """Error raised when integrator step fails reversibility check."""


class ConvergenceError(IntegratorError):
    """Error raised when solver fails to converge within allowed iterations."""


class LinAlgError(Error):
    """Error raised when a matrix operation raises a linear algebra error."""


class HamiltonianDivergenceError(IntegratorError):
    """Error raised when integration of Hamiltonian dynamics diverges."""


class AdaptationError(Error):
    """Error raised when adaptation of transition parameters fails."""


class ReadOnlyStateError(Error):
    """Error raised when writing to attributes of read-only chain state."""
