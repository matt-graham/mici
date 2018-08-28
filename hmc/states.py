"""Classes to represent states of Hamiltonian systems of various types."""

import copy


class SeparableHamiltonianState(object):
    """State of a separable Hamiltonian system.

    As well as recording the position and momentum state values and integration
    direction binary indicator, the state object is also used to cache derived
    quantities such as the potential and kinetic energy values and gradients
    for the current position and momentum values to avoid recalculation when
    these values are reused.

    Here separable means that the Hamiltonian can be expressed as the sum of
    a term depending only on the position (target) variables, typically denoted
    the potential energy, and a second term depending only on the momentum
    variables, typically denoted the kinetic energy.
    """

    def __init__(self, pos, mom, direction=1,
                 pot_energy_val=None, pot_energy_grad=None,
                 kin_energy_val=None, kin_energy_grad=None):
        self._pos = pos
        self._mom = mom
        self.direction = direction
        self.pot_energy_val = pot_energy_val
        self.pot_energy_grad = pot_energy_grad
        self.kin_energy_val = kin_energy_val
        self.kin_energy_grad = kin_energy_grad

    @property
    def n_dim(self):
        return self.pos.shape[0]

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.pot_energy_val = None
        self.pot_energy_grad = None

    @property
    def mom(self):
        return self._mom

    @mom.setter
    def mom(self, value):
        self._mom = value
        self.kin_energy_val = None
        self.kin_energy_grad = None

    def deep_copy(self):
        return copy.deepcopy(self)

    def copy(self):
        return copy.copy(self)

    def __str__(self):
        return '(\n  pos={0},\n  mom={1}\n)'.format(self.pos, self.mom)

    def __repr__(self):
        return 'SeparableHamiltonianState' + str(self)
