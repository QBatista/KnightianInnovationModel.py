"""
A module for representing households in the Knightian innovation model.

"""

import numpy as np
from knightian_model.utilities import utility_function_factory
from ._grid import create_P


class Household():
    """
    A class for representing discrete households in the Knightian model of
    innovation.

    Parameters
    ----------
    α : ndarray(float, ndim=1), optional(default=np.array([1.]))
        Distribution of households over perceived probability distributions
        of inventions being successful.

    ν : scalar(float), optional(default=2.)
        Coefficient of relative risk aversion.

    β : scalar(float), optional(default=0.9)
        Discount factor. Must be strictly less than 1.

    δ_vals : ndarray(float, ndim=1), optional(default=np.array([0.9, 0.95]))
        Grid for the depreciation rate variable δ.

    P_δ : ndarray(float, ndim=1), optional(default=np.array([0.5, 0.5]))
        Probability distribution over the values of δ.

    ζ_vals : ndarray(float, ndim=1), optional(default=np.array([1., 5.]))
        Grid for the labor income shock variable ζ.

    P_ζ : ndarray(float, ndim=2), optional(default=np.array([[0.5, 0.5], [0.5, 0.5]]))
        Markov transition matrix for ζ.

    π : ndarray(float, ndim=1), optional(default=np.array([0.5]))
        Perceived probabilities of an invention being successful.

    μ : scalar(float), optional(default=0.5)
        Arrival rate of invention opportunities.

    w_vals : ndarray(float, ndim=1), optional(default=None)
        Array containing the approximation nodes for the wealth state variable
        w. If None, it is automatically constructed.

    b_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for the borrowing choice
        variable b. If None, it is automatically constructed.

    k_tilde_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for both the state and
        choice net investment variable k_tilde. If None, it is automatically
        constructed.

    Attributes
    ----------
    α, ν, β, δ_vals, P_δ, ζ_vals, P_ζ, π, w_vals, b_vals, k_tilde_vals : See Parameters

    ι_vals : ndarray(float, ndim=1)
        Grid for the invention opportunity variable ι.

    u : callable
        A JIT-compiled utility function that takes consumption `c` as an input
        and returns the corresponding CRRA utility.

    P : ndarray(float, ndim=3)
        Joint probability distribution over the values of δ, ζ and ι.
        Probabilities vary by δ on the first axis, by ζ on the second axis,
        and by ι on the third axis.

    """

    def __init__(self, α=np.array([1.]), ν=2., β=0.9,
                 δ_vals=np.array([0.9, 0.95]), P_δ=np.array([0.5, 0.5]),
                 ζ_vals=np.array([1., 5.]),
                 P_ζ=np.array([[0.5, 0.5], [0.5, 0.5]]),
                 π=np.array([0.5]), μ=0.5, w_vals=None,  b_vals=None,
                 k_tilde_vals=None):

        # Check if parameters are valid
        self._check_invalid_α(α)
        self._check_invalid_β(β)

        if δ_vals.size != P_δ.size:
            raise ValueError('The dimensions of `δ_vals` and `P_δ` do not ' +
                             'match. Please change the values appropriately.')

        if (P_δ < 0.).any() or (P_δ > 1.).any() or P_δ.sum() != 1.:
            raise ValueError('P_δ is not a valid probability distribution.')

        if ζ_vals.size != P_ζ.shape[0]:
            raise ValueError('The dimensions of `ζ_vals` and `P_ζ` do not ' +
                             'match. Please change the values appropriately.')

        if P_ζ.shape[0] != P_ζ.shape[1]:
            raise ValueError('P_ζ must be a square matrix.')

        self._check_invalid_π(π)

        # Initialize parameters
        self._α = α
        self._ν = ν
        self._u = utility_function_factory(self.ν)  # Checks that ν is valid
        self._β = β
        self._δ_vals = δ_vals
        self._P_δ = P_δ
        self._ζ_vals = ζ_vals
        self._P_ζ = P_ζ
        self._ι_vals = np.array([0, 1])
        self._π = π
        self._μ = μ
        self._P_ι = np.array([1 - μ, μ])

        # Default grid (not used if all grids are specified by the user)
        b_lim = -30.
        a_max = 150.
        α_grid = 0.4
        num = 2 ** 9
        default_grid = b_lim + (a_max - b_lim) * \
            np.linspace(0, 1, num=num) ** (1 / α_grid)

        if w_vals is None:
            self._w_vals = default_grid
        else:
            self._w_vals = w_vals

        if b_vals is None:
            self._b_vals = default_grid
        else:
            self._b_vals = b_vals

        if k_tilde_vals is None:
            self._k_tilde_vals = default_grid
        else:
            self._k_tilde_vals = k_tilde_vals

        self._P = create_P(self.P_δ, self.P_ζ, self._P_ι)

    def __repr__(self):
        out = \
        """
        Household parameters
        --------------------------------
        Distribution of households α = %s
        Coefficient of relative risk aversion ν = %s
        Discount factor β = %s
        Distribution of perceived probabilities invention success of π = %s
        Arrival rate of invention opportunities μ = %s
        Labor income shocks ζ_vals = %s
        Depreciation shocks δ_vals = %s

        """ % (self.α, self.ν, self.β, self.π, self.μ, self.ζ_vals,
               self.δ_vals)

        return out

    def _check_invalid_α(self, value):
        "Raise a `ValueError` if the value of α is invalid"

        if (value < 0.).any() or (value > 1.).any() or value.sum() != 1.:
            raise ValueError('α is not a valid probability distribution.')

    @property
    def α(self):
        "Get the current value of α."

        return self._α

    @α.setter
    def α(self, new_value):
        "Set `new_value` as the new value for α."

        self._check_invalid_α(new_value)
        self._α = value

    @property
    def ν(self):
        "Get the current value of ν."

        return self._ν

    @ν.setter
    def ν(self, new_value):
        "Set `new_value` as the new value for ν."

        # Update utility function and check that ν is valid
        self._u = utility_function_factory(new_value)
        self._ν = new_value

    def _check_invalid_β(self, value):
        "Raise a `ValueError` if the value of β is invalid"

        if value > 1. or value < 0.:
            raise ValueError('β must be between 0 and 1.')

    @property
    def β(self):
        "Get the current value of β."

        return self._β

    @β.setter
    def β(self, new_value):
        "Set `new_value` as the new value for β."

        self._check_invalid_β(new_value)
        self._β = value

    @property
    def u(self):
        "Get the current utility function u."

        return self._u

    @property
    def δ_vals(self):
        "Get the current value of δ_vals."

        return self._δ_vals

    @property
    def P_δ(self):
        "Get the current value of P_δ."

        return self._P_δ

    @property
    def ζ_vals(self):
        "Get the current value of ζ_vals."

        return self._ζ_vals

    @property
    def P_ζ(self):
        "Get the current value of P_ζ."

        return self._P_ζ

    @property
    def ι_vals(self):
        "Get the current value of ι_vals."

        return self._ι_vals

    def _check_invalid_π(self, value):
        "Raise a `ValueError` if the value of π is invalid"

        for π in value:
            if π > 1. or π < 0.:
                raise ValueError('π must be between 0 and 1.')

    @property
    def π(self):
        "Get the current value of π."

        return self._π

    @π.setter
    def π(self, new_value):
        "Set `new_value` as the new value for π."

        _check_invalid_π(new_value)
        self._π = value

    def _check_invalid_μ(self, value):
        "Raise a `ValueError` if the value of μ is invalid"

        if value > 1. or value < 0.:
            raise ValueError('μ must be between 0 and 1.')

    @property
    def μ(self):
        "Get the current value of μ."

        return self._μ

    @μ.setter
    def μ(self, new_value):
        "Set `new_value` as the new value for π."

        _check_invalid_μ(new_value)
        self._μ = new_value

    @property
    def w_vals(self):
        "Get the current value of w_vals."

        return self._w_vals

    @w_vals.setter
    def w_vals(self, new_value):
        "Set `new_value` as the new value for w_vals."

        self._w_vals = new_value

    @property
    def b_vals(self):
        "Get the current value of b_vals."

        return self._b_vals

    @b_vals.setter
    def b_vals(self, new_value):
        "Set `new_value` as the new value for b_vals."

        self._b_vals = new_value

    @property
    def k_tilde_vals(self):
        "Get the current value of k_tilde_vals."

        return self._k_tilde_vals

    @k_tilde_vals.setter
    def k_tilde_vals(self, new_value):
        "Set `new_value` as the new value for k_tilde_vals."

        self._k_tilde_vals = new_value

    @property
    def P(self):
        "Get the current value of P."

        return self._P
