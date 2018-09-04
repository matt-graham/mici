"""Classes to represent states of Hamiltonian systems of various types."""

import copy


class BaseHamiltonianState(object):
    """Base class for state of a separable Hamiltonian system.

    As well as recording the position and momentum state values and integration
    direction binary indicator, the state object is also used to cache derived
    quantities such as (components of) the Hamiltonian function and gradients
    for the current position and momentum values to avoid recalculation when
    these values are reused.
    """

    def __init__(self, pos, mom, direction=1):
        self._pos = pos
        self._mom = mom
        self.direction = direction

    @property
    def n_dim(self):
        return self.pos.shape[0]

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        raise NotImplementedError()

    @property
    def mom(self):
        return self._mom

    @mom.setter
    def mom(self, value):
        raise NotImplementedError()

    def deep_copy(self):
        return copy.deepcopy(self)

    def copy(self):
        return copy.copy(self)

    def __str__(self):
        return '(\n  pos={0},\n  mom={1}\n)'.format(self.pos, self.mom)

    def __repr__(self):
        return type(self).__name__ + str(self)


class SeparableHamiltonianState(BaseHamiltonianState):
    """State of a separable Hamiltonian system.

    Here separable means that the Hamiltonian can be expressed as the sum of
    a term depending only on the position (target) variables, typically denoted
    the potential energy, and a second term depending only on the momentum
    variables, typically denoted the kinetic energy.
    """

    def __init__(self, pos, mom, direction=1, pot_energy=None,
                 grad_pot_energy=None, kin_energy=None, grad_kin_energy=None):
        super().__init__(pos, mom, direction)
        self.pot_energy = pot_energy
        self.grad_pot_energy = grad_pot_energy
        self.kin_energy = kin_energy
        self.grad_kin_energy = grad_kin_energy

    @BaseHamiltonianState.pos.setter
    def pos(self, value):
        self._pos = value
        self.pot_energy = None
        self.grad_pot_energy = None

    @BaseHamiltonianState.mom.setter
    def mom(self, value):
        self._mom = value
        self.kin_energy = None
        self.grad_kin_energy = None


class RiemannianHamiltonianState(BaseHamiltonianState):
    """State of a Riemmanian manifold Hamiltonian system."""

    def __init__(self, pos, mom, direction=1, pot_energy=None,
                 grad_pot_energy=None, metric=None, chol_metric=None,
                 inv_metric=None, inv_metric_mom=None, vjp_metric=None,
                 vjp_chol_metric=None, inv_chol_metric=None):
        super().__init__(pos, mom, direction)
        self.pot_energy = pot_energy
        self.grad_pot_energy = grad_pot_energy
        self.metric = metric
        self.chol_metric = chol_metric
        self.inv_metric = inv_metric
        self.inv_metric_mom = inv_metric_mom
        self.vjp_metric = vjp_metric
        self.inv_chol_metric = inv_chol_metric
        self.vjp_chol_metric = vjp_chol_metric

    @BaseHamiltonianState.pos.setter
    def pos(self, value):
        self._pos = value
        self.pot_energy = None
        self.grad_pot_energy = None
        self.metric = None
        self.chol_metric = None
        self.inv_metric = None
        self.inv_metric_mom = None
        self.vjp_metric = None
        self.inv_chol_metric = None
        self.vjp_chol_metric = None

    @BaseHamiltonianState.mom.setter
    def mom(self, value):
        self._mom = value
        self.inv_metric_mom = None
