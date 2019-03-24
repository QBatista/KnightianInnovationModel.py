"""
A module for solving the Knightian model of innovation.

"""

import numpy as np
from .function_factories import (utility_function_factory,
    production_function_factory)
from .helper_functions import (create_P, create_next_w, create_uc_grid,
    initialize_values_and_policies, compute_policy_grid)
from .household_dp_solvers import solve_dp_vi
import matplotlib.pyplot as plt
import warnings


# TODO:
# - Add tests

class KIMHouseholds():
    """
    A class for representing households in the Knightian model of innovation.

    Parameters
    ----------
    α : scalar(float), optional(default=1.)
        Measure of bold households.

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

    π : scalar(float), optional(default=1.)
        Probability of an invention being successful.

    w_vals : ndarray(float, ndim=1), optional(default=None)
        Array containing the approximation nodes for the wealth state variable
        w. If None, `np.linspace(1e-8, 50., 2**9)` is used.

    b_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for the borrowing choice
        variable b. If None, `np.linspace(-10., 10., 2**9)` is used.

    k_tilde_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for both the state and
        choice net investment variable k_tilde. If None,
        `np.linspace(1e-8, 50., 2**9)` is used.

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

    def __init__(self, α=1., ν=2., β=0.9, δ_vals=np.array([0.9, 0.95]),
                 P_δ=np.array([0.5, 0.5]), ζ_vals=np.array([1., 5.]),
                 P_ζ=np.array([[0.5, 0.5], [0.5, 0.5]]), π=1., w_vals=None,
                 b_vals=None, k_tilde_vals=None):

        # Parameters check
        self._check_invalid_α(α)
        self._check_invalid_β(β)

        if δ_vals.size != P_δ.size:
            raise ValueError('The dimensions of `δ_vals` and `P_δ` do not ' +
                             'match. Please change the values appropriately.')

        if (P_δ < 0.).any() or (P_δ > 1.).any() or P_δ.sum() != 1.:
            raise ValueError('The new value of P_δ is not a valid ' +
                             'probability distribution.')

        if ζ_vals.size != P_ζ.shape[0]:
            raise ValueError('The dimensions of `ζ_vals` and `P_ζ` do not ' +
                             'match. Please change the values appropriately.')

        if P_ζ.shape[0] != P_ζ.shape[1]:
            raise ValueError('P_ζ must be a square matrix')

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
        self._P_ι = np.array([1 - π, π])

        if w_vals is None:
            self._w_vals = np.linspace(1e-8, 50., 2**9)
        else:
            self._w_vals = w_vals

        if b_vals is None:
            self._b_vals = np.linspace(-10., 10., 2**9)
        else:
            self._b_vals = b_vals

        if k_tilde_vals is None:
            self._k_tilde_vals = np.linspace(1e-8, 50., 2**9)
        else:
            self._k_tilde_vals = k_tilde_vals

        self._P = create_P(self.P_δ, self.P_ζ, self._P_ι)

    def __repr__(self):
        out = \
        """
        Household parameters
        --------------------------------
        Measure of bold households α = %s
        Coefficient of relative risk aversion ν = %s
        Discount factor β = %s
        Inventing success rate π = %s

        """ % (self.α, self.ν, self.β, self.π)

        return out

    def _check_invalid_α(self, value):
        "Raise a `ValueError` if the value of α is invalid"

        if value > 1. or value < 0.:
            raise ValueError('α must be between 0 and 1.')

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

        if value > 1. or value < 0.:
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
        self._P = create_P(self.P_δ, self.P_ζ, self._P_ι)

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


class KIMFirms():
    """
    A class for representing firms in the Knightian model of innovation.

    Parameters
    ----------
    A : scalar(float), optional(default=3.)
        Scale of production parameter.

    σ_1 : scalar(float), optional(default=0.3)
        Capital share of production.

    σ_2 : scalar(float), optional(default=0.3)
        Labor share of production.

    Attributes
    ----------
    A, σ_1, σ_2 : See Parameters

    F : callable
        A JIT-compiled utility function that takes capital `K`, labor `L` and
        intermediate goods `M` as inputs and returns the corresponding output.

    F_K : callable
        The JIT-compiled derivative function of `F` with respect to `K`.

    F_L : callable
        The JIT-compiled derivative function of `F` with respect to `L`.

    F_M : callable
        The JIT-compiled derivative function of `F` with respect to `M`.

    """

    def __init__(self, A=3., σ_1=0.3, σ_2=0.3):
        # The following also checks that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(A, σ_1, σ_2)

        self._A = A
        self._σ_1 = σ_1
        self._σ_2 = σ_2


    def __repr__(self):
        out = \
        """
        Firm parameters
        --------------------------------
        Scale of production parameter A = %s
        Capital share parameter σ_1 = %s
        Labor share parameter σ_2 = %s

        """ % (self.A, self.σ_1, self.σ_2)

        return out

    @property
    def A(self):
        "Get the current value of A."

        return self._A

    @A.setter
    def A(self, value):
        # Update production functions and check that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(value, self.σ_1, self.σ_2)
        self._A = A

    @property
    def σ_1(self):
        "Get the current value of σ_1."

        return self._σ_1

    @σ_1.setter
    def σ_1(self, value):
        # Update production functions and check that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(self.A, value, self.σ_2)
        self._σ_1 = value

    @property
    def σ_2(self):
        "Get the current value of σ_2."

        return self._σ_2

    @σ_2.setter
    def σ_2(self, value):
        # Update production functions and check that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(self.A, self.σ_1, value)
        self._σ_1 = value

    @property
    def F(self):
        "Get the current production function F."

        return self._F

    @property
    def F_K(self):
        """
        Get the current derivative function of the production function F with
        respect to K.
        """

        return self._F_K

    @property
    def F_L(self):
        """
        Get the current derivative function of the production function F with
        respect to L.
        """

        return self._F_L

    @property
    def F_M(self):
        """
        Get the current derivative function of the production function F with
        respect to M.
        """

        return self._F_M


class KnightianInnovationModel():
    """
    A class for representing the Knightian model of innovation.

    Parameters
    ----------
    households : object
        Instance of KIMHouseholds representing the households in the model.

    firms : object
        Instance of KIMFirms representing the firms in the model.

    γ : scalar(float), optional(default=0.95)
        Obsolescence rate of intermediate good techonologies.

    K : scalar(float), optional(default=11.)
        Aggregate capital stock.

    L : scalar(float), optional(default=1.)
        Aggregate labor supply.

    M : scalar(float), optional(default=1.)
        Aggregate intermediary good.

    R : scalar(float), optional(default=1.02)
        Gross risk-free interest rate.

    Attributes
    ----------
    hh : object
        See `households` parameter.

    firms, K, L, M, R, γ : See Parameters

    r : scalar(float)
        Equilibrium risky interest rate computed as the marginal product of the
        production function with respect to capital evaluated at `K`, `L` and
        `M`.

    wage : scalar(float)
        Equilibrium wage computed as the marginal product of the production
        function with respect to labor evaluated at `K`, `L` and `M`.

    p_M : scalar(float)
        Equilibrium price of the intermediary good computed as the marginal
        product of the production function with respect to the intermediary
        good evaluated at `K`, `L` and `M`.

    V1_star : ndarray(float, ndim=3)
        Array containing estimates of the first optimal value function.

    V2_star : ndarray(float, ndim=3)
        Array containing estimates of the second optimal value function.

    π_star : ndarray(float, ndim=4)
        Array containing estimates of the optimal policy function.

    b_av : ndarray(float, ndim=4)
        Array containing estimates of the action values for the borrowing
        choice variable at approximation nodes `hh.b_vals`.

    k_tilde_av : ndarray(float, ndim=4)
        Array containing the estimates of the action values for different net
        investment levels at the approximation nodes `hh.k_tilde_vals`.

    """

    def __init__(self, households, firms, γ=0.95, K=11., L=1., M=1., R=1.02):
        self._hh = households
        self._firms = firms
        self._K = K
        self._L = L
        self._M = M
        self._R = R
        self._γ = γ

        self._compute_params()

        (self._V1_star, self._V1_store, self._V2_star, self._V2_store, self._b_av,
         self._k_tilde_av, self._π_star) = \
            initialize_values_and_policies(self._states_vals,
                                           self.hh.b_vals)

    def __repr__(self):
        self._params = \
        """
        Model parameters
        --------------------------------
        Aggregate capital stock K = %s
        Aggregate labor supply L = %s
        Aggregate intermediary good M = %s
        Net risky interest rate r = %s
        Gross risk-free interest rate R = %s
        Wage `wage` = %s
        Present value of a new intermediate goods invention Γ_star = %s
        Oldest technology j_bar = %s
        """ % (self.K, self.L, self.M, self.r, self.R, self.wage, self.Γ_star,
               self.j_bar)

        return self.hh.__repr__() + self.firms.__repr__() + self._params

    @property
    def hh(self):
        return self._hh

    @property
    def firms(self):
        return self._firms

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, new_value):
        self._K = new_value
        self._compute_params()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, new_value):
        self._L = new_value
        self._compute_params()

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, new_value):
        self._M = new_value
        self._compute_params()

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, new_value):
        self._R = new_value
        self._compute_params()

    @property
    def γ(self):
        return self._γ

    @γ.setter
    def γ(self, new_value):
        self._γ = new_value
        self._compute_params()

    @property
    def r(self):
        return self._r

    @property
    def wage(self):
        return self._wage

    @property
    def p_M(self):
        return self._p_M

    @property
    def Γ_star(self):
        return self._Γ_star

    @property
    def j_bar(self):
        return self._j_bar

    @property
    def V1_star(self):
        return self._V1_star

    @property
    def V2_star(self):
        return self._V2_star

    @property
    def π_star(self):
        return self._π_star

    @property
    def b_av(self):
        return self._b_av

    @property
    def k_tilde_av(self):
        return self._k_tilde_av

    def _compute_params(self):
        # Compute prices
        self._r = self.firms.F_K(self.K, self.L, self.M)
        self._wage = self.firms.F_L(self.K, self.L, self.M)
        self._p_M = self.firms.F_M(self.K, self.L, self.M)

        # Compute `j_bar` and `Γ_star`
        self._j_bar = np.floor(np.log(self.R / self.p_M) / np.log(self.γ))
        js = np.arange(0, self._j_bar + 1)
        self._Γ_star = ((self.γ ** js * self.p_M / self.R - 1) * \
            self.R ** (-js)).sum()

        # Compute states
        self.hh._next_w, self.hh._next_w_star = \
            create_next_w(self.r, self.hh.δ_vals,
                          self.hh.k_tilde_vals,
                          self.hh.b_vals, self.R, self.Γ_star)

        self._states_vals = (self.hh.w_vals, self.hh.ζ_vals, self.hh.ι_vals,
                             self.hh.k_tilde_vals)

        self._check_params()

    def _check_params(self):
        "Sanity check for the parameters of the model"

        w_max = self.hh.w_vals.max()

        if self.Γ_star > w_max:
            warning.warn('w_max is lower than Γ_star: this is problematic' +
                         'for extrapolation.', category=RuntimeWarning)

        if self.Γ_star / w_max > 0.1:
            warnings.warn('Γ_star is high relative to the maximum wealth:' +
                          'this might result in a low quality extrapolation.',
                          category=RuntimeWarning)

        if self.R - 1 > self.r:
            warnings.warn('Risk-free rate is higher than risky return.',
                          category=RuntimeWarning)

        if self.Γ_star <= 0:
            warnings.warn('Γ_star is not positive.', category=RuntimeWarning)

        if self.wage <= 0:
            warnings.warn('Wage is not positive.', category=RuntimeWarning)

        if self.p_M <= 0:
            warnings.warn('p_M is not positive.', category=RuntimeWarning)

        if self.hh.ζ_vals.min() * self.wage >= self.Γ_star:
            warnings.warn('Γ_star is lower than the minimum earnings from' +
                          'labor', category=RuntimeWarning)

        if (self.hh.δ_vals * (1 + self.r)).mean() <= self.R:
            warnings.warn('Expected value of (1+r) is less than R.',
                          category=RuntimeWarning)

        if (self.hh.δ_vals * (1 + self.r)).mean() * self.hh.β >= 1.:
            warnings.warn('Expected value of (1+r) * β is greater or equal' +
                          'to 1.', category=RuntimeWarning)

    def solve_household_DP_problem(self, method=0, tol=1e-7):
        if method == 0:
            uc = create_uc_grid(self.hh.u, self._states_vals, self.wage)

            method_args = \
                (self.hh.P, uc, self.hh.b_vals, self._k_tilde_av, self.b_av,
                 self.hh._next_w_star, self.hh._next_w)

            results = solve_dp_vi(self._V1_star, self._V1_store, self.V2_star,
                                  self._V2_store, self._states_vals,
                                  self.hh.δ_vals,
                                  self.hh.π, self.hh.β, method, method_args,
                                  tol=tol)

            compute_policy_grid(self._π_star, self._V1_star, self._b_av,
                                self.hh.b_vals, self._k_tilde_av,
                                self.hh.k_tilde_vals)

    def plot_value_functions(self, markersize=1.5):
        """
        Plot of the value functions.

        Parameters
        ----------
        markersize : scalar(float), optional(default=1.5)
            Marker size of the points in the plot.

        """

        plt.figure(figsize=(14, 12))

        for ι in self.hh.ι_vals:
            plt.subplot(2, 2, ι + 1)
            for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                plt.plot(self.hh.w_vals, self.V1_star[ι, ζ_i, :],
                         'o', label=r'$ζ$ = ' + str(ζ), markersize=markersize)
                plt.title(r'$V_1$($ω$, $ζ$, $ι$=' + str(ι) + ')')

            plt.ylabel(r'$V$')
            plt.xlabel(r'$ω$')
            plt.legend()

            plt.subplot(2, 2, 3 + ι)
            for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                plt.plot(self.hh.k_tilde_vals, self.V2_star[ζ_i, :, ι],
                         'o', label=r'$ζ$ = ' + str(ζ), markersize=markersize)
                plt.title(r'$V_2$($\tilde{k}$, $ζ$, $ι$=' + str(ι) + ')')

            plt.ylabel(r'$V$')
            plt.xlabel(r'$\tilde{k}$')
            plt.legend()

        plt.suptitle('Value Function Plots', fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.show()

    def plot_policy_function(self, markersize=1.5):
        """
        Plot of the policy function.

        Parameters
        ----------
        markersize : scalar(float), optional(default=1.5)
            Marker size of the points in the plot.

        """

        # This function takes a little bit of time to run -- is there a way
        # to make it faster?
        # Also: refactor code to make it less repetitive

        plt.figure(figsize=(14, 36))
        cnt = 1

        for ι in self.hh.ι_vals:
            for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                plt.subplot(6, 2, cnt)
                plt.plot(self.hh.w_vals, self.π_star[ι, ζ_i, :, 0], 'o',
                         label=r'$ζ$ = ' + str(ζ), markersize=markersize)

                plt.title(r'Invention choice variable ι_D when $ζ$ = ' +
                          str(ζ) + ' and $ι$ = ' + str(ι))
                plt.ylabel(r'$ι_D$')
                plt.xlabel(r'$ω$')

                cnt += 1

        for ι in self.hh.ι_vals:
            plt.subplot(6, 2, cnt)
            for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                plt.plot(self.hh.w_vals, self.π_star[ι, ζ_i, :, 1], 'o',
                         label=r'$ζ$ = ' + str(ζ), markersize=markersize)

            plt.title(r'Net investment choice variable $\tilde{k}$ when $ι$' +
                      ' = ' + str(ι))
            plt.ylabel(r'$\tilde{k}$')
            plt.xlabel(r'$ω$')
            plt.legend()

            cnt += 1

        for ι in self.hh.ι_vals:
            plt.subplot(6, 2, cnt)
            for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                plt.plot(self.hh.w_vals, self.π_star[ι, ζ_i, :, 2], 'o',
                         label=r'$ζ$ = ' + str(ζ), markersize=markersize)

            plt.title(r'Borrowing choice variable $b$ when $ι$ = ' + str(ι))
            plt.ylabel(r'$b$')
            plt.xlabel(r'$ω$')
            plt.legend()

            cnt += 1

        for ι in self.hh.ι_vals:
            plt.subplot(6, 2, cnt)
            for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                k = self.π_star[ι, ζ_i, :, 1] - self.π_star[ι, ζ_i, :, 2]
                plt.plot(self.hh.w_vals, k, 'o', label=r'$ζ$ = ' + str(ζ),
                         markersize=markersize)

            plt.title(r'Gross investment choice variable $k$ when $ι$ = ' +
                      str(ι))
            plt.ylabel(r'$k$')
            plt.xlabel(r'$ω$')
            plt.legend()

            cnt += 1

        for ι in self.hh.ι_vals:
            plt.subplot(6, 2, cnt)
            for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                c = (1 - self.π_star[ι, ζ_i, :, 0]) * ζ * self.wage + \
                    self.hh.w_vals - self.π_star[ι, ζ_i, :, 1]
                plt.plot(self.hh.w_vals, c, 'o', label=r'$ζ$ = ' + str(ζ),
                         markersize=markersize)

            plt.title(r'Consumption choice variable $c$ when $ι$ = ' + str(ι))
            plt.ylabel(r'$c$')
            plt.xlabel(r'$ω$')
            plt.legend()

            cnt += 1

        plt.suptitle('Policy Function Plots', fontsize=16)
        plt.subplots_adjust(top=0.95)
        plt.show()

    def plot_stationary_distribution():
        pass

    def compute_stationary_distribution(self):
        pass

    def solve(self):
        pass
