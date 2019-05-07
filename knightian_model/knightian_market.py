"""
A module for representing the market in the Knightian innovation model

"""

import numpy as np
from scipy.stats import gaussian_kde
from knightian_model import KIMHouseholds, KIMFirms
from .helper_functions import (create_next_w, create_uc_grid,
    initialize_values_and_policies, compute_policy_grid)
from .household_dp_solvers import solve_dp_vi
from .stationary_distribution import MC
import matplotlib.pyplot as plt
import warnings


# TODO(QBatista): Implement initial guess for V_star

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

        # Check if parameters are valid
        self._check_invalid_K(K)
        self._check_invalid_L(L)
        self._check_invalid_M(M)
        self._check_invalid_γ(γ)

        # Initialize parameters

        self._hh = households
        self._firms = firms
        self._K = K
        self._L = L
        self._M = M
        self._R = R
        self._γ = γ

        self._compute_params()

        (self._V1_star, self._V1_store, self._V2_star, self._V2_store,
         self._b_av, self._k_tilde_av, self._π_star) = \
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

    def _check_invalid_K(self, value):
        "Raise a `ValueError` if the value for K is invalid"

        if value <= 0.:
            raise ValueError('K must be positive.' )

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, new_value):
        self._check_invalid_K(new_value)
        self._K = new_value
        self._compute_params()

    def _check_invalid_L(self, value):
        "Raise a `ValueError` if the value for L is invalid"

        if value <= 0.:
            raise ValueError('L must be positive.')

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, new_value):
        self._check_invalid_L(new_value)
        self._L = new_value
        self._compute_params()

    def _check_invalid_M(self, value):
        "Raise a `ValueError` if the value for M is invalid"

        if value <= 0.:
            raise ValueError('M must be positive.')

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, new_value):
        self._check_invalid_M(new_value)
        self._M = new_value
        self._compute_params()

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, new_value):
        self._R = new_value
        self._compute_params()

    def _check_invalid_γ(self, value):
        "Raise a `ValueError` if the value for γ is invalid"

        if value <= 0.:
            raise ValueError('γ must be positive.')

    @property
    def γ(self):
        return self._γ

    @γ.setter
    def γ(self, new_value):
        self._check_invalid_γ(new_value)
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
            warnings.warn('w_max is lower than Γ_star.',
                         category=RuntimeWarning)

        if self.Γ_star / w_max > 0.1:
            warnings.warn('Γ_star is high relative to the maximum wealth.',
                          category=RuntimeWarning)

        if self.R - 1 > self.r:
            warnings.warn('The risk-free rate is higher than risky rate.',
                          category=RuntimeWarning)

        if self.Γ_star <= 0:
            warnings.warn('Γ_star is not positive.', category=RuntimeWarning)

        if self.wage <= 0:
            warnings.warn('Wage is not positive.', category=RuntimeWarning)

        if self.p_M <= 0:
            warnings.warn('p_M is not positive.', category=RuntimeWarning)

        if self.hh.ζ_vals.min() * self.wage >= self.Γ_star:
            warnings.warn('Γ_star is lower than the minimum earnings from ' +
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

    def plot_stationary_distribution(self, popu, pdfs):

        for ζ_i, ζ_val in enumerate(self.hh.ζ_vals):

            pdf = pdfs[ζ_i]
            sub_popu = popu[popu[:, 1] == ζ_i, :]

            # histgram of population distribution over w
            plt.hist(sub_popu[:, 0], bins=50, density=True, label="sample")

            # the kernel density fit
            w_max = max(sub_popu[:, 0])
            w_min = min(sub_popu[:, 0])
            plt.plot(np.linspace(w_min, w_max, 100),
                     pdf(np.linspace(w_min, w_max, 100)),
                     label="kernel density fit")

            plt.title(f"stationary distribution P(w, ζ={ζ_val})")
            plt.xlabel("w")
            plt.ylabel(f"P(w, ζ={ζ_val})")
            plt.legend()
            plt.show()

    def compute_stationary_distribution(self, N=10000):
        """
        set equal intial population for different ζ values.
        each subgroup has population size N.
        """

        # initialize the population
        popu = np.empty((2 * N, 2))

        w_min, w_max = min(self.hh.w_vals), max(self.hh.w_vals)
        popu[:N, 0] = np.linspace(w_min, w_max, N)
        popu[:N, 1] = 0
        popu[N:, 0] = np.linspace(w_min, w_max, N)
        popu[N:, 1] = 1

        # Monte Carlo Simulation
        # set μ = 0.5 for now
        # need to change it to self.hh.μ
        MC(popu, self.π_star, self.hh.w_vals, self.hh.ζ_vals,
           self.hh.δ_vals, self.Γ_star, self.hh.P_ζ, self.hh.P_δ,
           0.5, self.hh.π, self.r, self.R)

        # kernel density fit
        pdfs = [gaussian_kde(popu[popu[:, 1] == ζ_i, 0])
                for ζ_i in range(len(self.hh.ζ_vals))]

        return popu, pdfs

    def solve(self):
        raise NotImplementedError("Coming soon.")
