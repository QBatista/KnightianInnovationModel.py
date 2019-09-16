"""
A module for representing the market in the Knightian innovation model

"""

import numpy as np
from interpolation import interp
from knightian_model.discrete import Household, Firm
from ._grid import (create_next_w, create_uc_grid, compute_policy_grid,
    initialize_values_and_policies)
from ._dp_solvers import solve_dp_vi
from ._stationary_distribution import MC
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings


# Plotting colors
colors = ['#1f77b4',  # muted blue
          '#ff7f0e',  # safety orange
          '#2ca02c',  # cooked asparagus green
          '#d62728',  # brick red
          '#9467bd',  # muted purple
          '#8c564b',  # chestnut brown
          '#e377c2',  # raspberry yogurt pink
          '#7f7f7f',  # middle gray
          '#bcbd22',  # curry yellow-green
          '#17becf'   # blue-teal
          ]

# TODO(QBatista):
# - Implement initial guess for V_star
# - Figure out how to best deal with the true probability of invention success
#   for MC simulation

class KnightianInnovationModel():
    """
    A class for representing the Knightian model of innovation in discrete
    time.

    Parameters
    ----------
    households : Households
        Instance of Households representing the households in the
        model.

    firms : object
        Instance of Firm representing the firms in the model.

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
    hh, firms, K, L, M, R, γ : See Parameters

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

    def __init__(self, households, firms, γ=0.95, K=9.10, L=0.775, M=1.15,
                 R=1.02):

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

        self.solution_data = [initialize_values_and_policies(self._states_vals,
                                                             self.hh.b_vals)
                              for i in range(households.α.size)]

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
            warnings.warn('Expected value of (1+r) * β is greater or equal ' +
                           'to 1.', category=RuntimeWarning)

    def solve_household_DP_problem(self, method=0, tol=1e-8):
        if method == 0:
            uc = create_uc_grid(self.hh.u, self._states_vals, self.wage)

            for i, π in enumerate(self.hh.π):
                V1_star, V1_store, V2_star, V2_store, b_av, k_tilde_av, π_star = \
                    self.solution_data[i]

                method_args = \
                    (self.hh.P, uc, self.hh.b_vals, k_tilde_av, b_av,
                    self.hh._next_w_star, self.hh._next_w)

                print("=" * 37)
                print('Solving problem of household', str(i+1), 'out of',
                      self.hh.π.size, '(π=' + str(π) + ')')
                print("=" * 37)

                results = solve_dp_vi(V1_star, V1_store, V2_star, V2_store,
                                      self._states_vals, self.hh.δ_vals, π,
                                      self.hh.β, method, method_args, tol=tol)

                print("-" * 35)
                print('Compute optimal policy...')
                compute_policy_grid(π_star, V1_star, b_av,
                                    self.hh.b_vals, k_tilde_av,
                                    self.hh.k_tilde_vals)
                print('...completed')

    def plot_value_function(self, height=1400, width=1400, renderer='browser',
                            markersize=1.5):
        """
        Plot the value function.

        Parameters
        ----------
        height : scalar(int), optional(default=1400)
            Height of the plot.

        width : scalar(int), optional(default=1400)
            Width of the plot.

        renderer: str or None, optional(default='browser')
            A string containing the names of one or more registered renderers
            (separated by '+' characters) or None.  If None, then the default
            renderers specified in plotly.io.renderers.default are used.

        markersize : scalar(float), optional(default=1.5)
            Marker size of the points in the plot.

        """
        subplot_titles = [r'$V_1(ω, ζ, ι=0; π)$',
                          r'$V_1(ω, ζ, ι=1; π)$',
                          r'$V_2(\tilde{k}, ζ, ι=0; π)$',
                          r'$V_2(\tilde{k}, ζ, ι=1; π)$']

        subplots = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles,
                                 specs=[[{}, {}],
                                        [{}, {}]])

        fig = go.Figure(subplots)
        mode = 'markers'

        for h_i, π in enumerate(self.hh.π):
            V1_star, V1_store, V2_star, V2_store, b_av, k_tilde_av, π_star = \
                self.solution_data[h_i]
            for ι in self.hh.ι_vals:
                col_nb = int(ι + 1)
                for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                    group = h_i * self.hh.ζ_vals.size + ζ_i
                    name = 'ζ=' + str(ζ) + ', π=' + str(π)
                    marker = dict(size=markersize, color=colors[ζ_i])
                    legendgroup = colors[group]
                    showlegend = bool(ι == 0)

                    fig.add_trace(go.Scatter(x=self.hh.w_vals,
                                             y=V1_star[ι, ζ_i, :],
                                             name=name,
                                             mode=mode,
                                             marker=marker,
                                             legendgroup=legendgroup,
                                             showlegend=showlegend),
                                  row=1, col=col_nb)

                    fig.add_trace(go.Scatter(x=self.hh.k_tilde_vals,
                                             y=V2_star[ζ_i, :, ι],
                                             name=name,
                                             mode=mode,
                                             marker=marker,
                                             legendgroup=legendgroup,
                                             showlegend=False),
                                  row=2, col=col_nb)

        fig.update_layout(height=height, width=width)

        fig.show(renderer)

    def plot_policy_function(self, height=3400, width=1400, renderer='browser',
                             markersize=1.5):
        """
        Plot the policy function.

        Parameters
        ----------
        height : scalar(int), optional(default=3400)
            Height of the plot.

        width : scalar(int), optional(default=1400)
            Width of the plot.

        renderer: str or None, optional(default='browser')
            A string containing the names of one or more registered renderers
            (separated by '+' characters) or None.  If None, then the default
            renderers specified in plotly.io.renderers.default are used.

        markersize : scalar(float), optional(default=1.5)
            Marker size of the points in the plot.

        """

        subplot_titles = ['Invention choice variable ι_D when ι=0',
                          'Invention choice variable ι_D when ι=1',
                          'Net investment choice variable when ι=0',
                          'Net investment choice variable when ι=1',
                          'Borrowing choice variable b when ι=0',
                          'Borrowing choice variable b when ι=1',
                          'Gross investment choice variable k when ι=0',
                          'Gross investment choice variable k when ι=1',
                          'Consumption choice variable c when ι=0',
                          'Consumption choice variable c when ι=1',
                          'Savings when ι=0',
                          'Savings when ι=1'
                         ]

        nb_rows = len(subplot_titles) // 2
        subplots = make_subplots(rows=nb_rows, cols=2,
                                 subplot_titles=subplot_titles,
                                 specs=[[{}, {}]] * nb_rows)

        fig = go.Figure(subplots)
        mode = 'markers'

        for h_i, π in enumerate(self.hh.π):
            V1_star, V1_store, V2_star, V2_store, b_av, k_tilde_av, π_star = \
                self.solution_data[h_i]
            k = π_star[:, :, :, 1] - π_star[:, :, :, 2]
            c = (1 - π_star[:, :, :, 0]) * self.hh.ζ_vals.reshape((1, -1, 1)) * self.wage + \
                        self.hh.w_vals - π_star[:, :, :, 1]
            s = π_star[:, :, :, 1] - self.hh.w_vals.reshape((1, 1, -1))

            data = [(self.hh.w_vals, π_star[:, :, :, 0]),
                    (self.hh.w_vals, π_star[:, :, :, 1]),
                    (self.hh.k_tilde_vals, π_star[:, :, :, 2]),
                    (self.hh.w_vals, k),
                    (self.hh.w_vals, c),
                    (self.hh.w_vals, s)]

            for ι in self.hh.ι_vals:
                col_nb = int(ι + 1)
                for ζ_i, ζ in enumerate(self.hh.ζ_vals):
                    for row_nb, (x, y) in enumerate(data):
                        group = h_i * self.hh.ζ_vals.size + ζ_i
                        name = r'ζ=' + str(ζ) + ', π=' + str(π)
                        showlegend = bool(ι == 0 and row_nb == 0)
                        legendgroup = colors[group]
                        marker = dict(size=markersize, color=colors[ζ_i])
                        fig.add_trace(go.Scatter(x=x,
                                                 y=y[ι, ζ_i, :],
                                                 name=name,
                                                 mode=mode,
                                                 marker=marker,
                                                 legendgroup=legendgroup,
                                                 showlegend=showlegend),
                                      row=row_nb+1, col=col_nb)

        fig.update_xaxes(title_text=r'$w$')
        fig.update_layout(height=height, width=width)

        fig.show(renderer)

    def compute_stationary_distribution(self, N=10000, maxiter=1000,
                                        tol=1e-5, verbose=True):
        """
        set equal intial population for different ζ values.
        each subgroup has population size N.
        """

        # initialize the population
        popu = np.empty((N, 2, self.hh.π.size))

        w_min, w_max = min(self.hh.w_vals), max(self.hh.w_vals)
        popu[:, 0] = np.linspace(w_min, w_max, N)[:, np.newaxis]
        popu[:, 1] = 0

        P_ζ_cdfs = self.hh.P_ζ.cumsum(axis=1)

        for h_i in range(self.hh.π.size):
            V1_star, V1_store, V2_star, V2_store, b_av, k_tilde_av, π_star = \
                self.solution_data[h_i]

            # Monte Carlo Simulation
            MC(popu[:, :, h_i], π_star, self.hh.w_vals, self.hh.ζ_vals,
               self.hh.δ_vals, self.Γ_star, P_ζ_cdfs, self.hh.P_δ,
               self.hh.μ, self.hh.π[0], self.r, self.R, maxiter=maxiter,
               tol=tol, verbose=verbose)

        return popu

    def plot_stationary_distribution(self, popu, bins=50, height=2000,
                                     width=1400, renderer='browser'):
        """
        Plot the stationary distribution using an histogram.

        Parameters
        ----------
        popu : ndarray(dtype=float)
            Samples from the stationary distribution.

        bins : scalar(int), optional(default=50)
            Number of bins for the histogram plot.

        height : scalar(int), optional(default=2000)
            Height of the plot.

        width : scalar(int), optional(default=1400)
            Width of the plot.

        renderer: str or None, optional(default='browser')
            A string containing the names of one or more registered renderers
            (separated by '+' characters) or None.  If None, then the default
            renderers specified in plotly.io.renderers.default are used.

        markersize : scalar(float), optional(default=1.5)
            Marker size of the points in the plot.

        """
        n = self.hh.ζ_vals.size * self.hh.π.size

        subplot_titles = ['Stationary Distribution for ζ=' + str(ζ_val) +
                          ', π=' + str(π) for ζ_val in self.hh.ζ_vals
                          for π in self.hh.π]

        subplots = make_subplots(rows=n, cols=1,
                                 subplot_titles=subplot_titles,
                                 specs=[[{}]] * n)

        fig = go.Figure(subplots)

        for h_i in range(self.hh.π.size):
            for ζ_i, ζ_val in enumerate(self.hh.ζ_vals):
                row_nb = h_i * self.hh.ζ_vals.size + ζ_i + 1
                sub_popu = popu[popu[:, 1, h_i] == ζ_i, 0, h_i]

                fig.add_trace(go.Histogram(x=sub_popu, histnorm='probability'),
                              row=row_nb, col=1)

        fig.update_layout(height=height, width=width)

        fig.show(renderer)

    def compute_aggregates(self, popu):
        """
        compute aggregates using stationary distribution.
        """

        # simplify notations
        π_star, w_vals, ζ_vals = self.π_star, self.hh.w_vals, self.hh.ζ_vals

        # K_tilde and B
        aggregates = np.zeros(2)
        for h_i in range(self.hh.π.size):
            for ζ_i in range(len(ζ_vals)):
                w_subsample = popu[popu[:, 1, h_i] == ζ_i, 0, h_i]
                ζ_weight = len(w_subsample) / len(popu)
                for ι_i in range(2):
                    # compute K_tilde and B
                    for i in range(2):
                        aggregates[i] += self.hh.α[h_i] * self.hh.P_ι[ι_i] * ζ_weight * \
                            interp(w_vals, π_star[ι_i, ζ_i, :, i+1],
                                   w_subsample).mean()

        # K = K_tilde - B
        K = aggregates[0] - aggregates[1]
        B = aggregates[1]

        return K, B

    def solve(self):
        raise NotImplementedError("Coming soon.")
