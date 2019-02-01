"""
Tests for solve_dp.py

"""

import numpy as np
from numba import njit
from ..household_dp_solvers import solve_dp_vi
from .wrappers import solve_dp_wrapper
from ..function_factories import (utility_function_factory,
    production_function_factory)


class TestSolveDP:
    def __init__(self, α=1, A=3., σ_1=0.3, σ_2=0.3, γ=0.95, μ=0.5, π=0.7,
                 β=0.9, ν=2, δ_vals=np.array([0.9, 0.95]),
                 P_δ=np.array([0.5, 0.5]), w_min=1e-10, w_max=200,
                 w_size=2**9, ζ_vals=np.array([1., 5.]),
                 P_ζ=np.array([[0.5, 0.5], [0.5, 0.5]]),
                 ι_vals=np.array([0, 1]), b_min=-10., b_max=0., b_size=2**9,
                 k_tilde_min=-10., k_tilde_max=200., k_tilde_size=2**9, K=11,
                 L=1, M=1, R=1.02):
        self.α = α
        self.A = A
        self.σ_1 = σ_1
        self.σ_2 = σ_2
        self.γ = γ
        self.μ = μ
        self.π = π
        self.β = β
        self.ν = ν
        self.δ_vals = δ_vals
        self.P_δ = P_δ
        self.w_min = w_min
        self.w_max = w_max
        self.w_size = w_size
        self.w_vals = np.linspace(self.w_min, self.w_max, self.w_size)
        self.ζ_vals = ζ_vals
        self.P_ζ = P_ζ
        self.ι_vals = ι_vals
        self.P_ι = np.array([1 - self.μ, self.μ])
        self.b_min = b_min
        self.b_max = b_max
        self.b_size = b_size
        self.b_vals = np.linspace(self.b_min, self.b_max, self.b_size)
        self.k_tilde_min = k_tilde_min
        self.k_tilde_max = k_tilde_max
        self.k_tilde_size = k_tilde_size
        self.k_tilde_vals = np.linspace(self.k_tilde_min, self.k_tilde_max,
                                        self.k_tilde_size)
        self.u = njit(utility_function_factory(self.ν))
        self.F, self.F_K, self.F_L, self.F_M = [njit(f) for f in
            production_function_factory(self.A, self.σ_1, self.σ_2)]
        self.K, self.L, self.M = K, L, M
        self.R = R
        self.r = self.F_K(self.K, self.L, self.M)
        self.wage = self.F_L(self.K, self.L, self.M)
        self.p_M = self.F_M(self.K, self.L, self.M)
        self.j_bar = np.floor(np.log(self.R / self.p_M) / np.log(self.γ))
        self.js = np.arange(0, self.j_bar + 1)
        self.Γ_star = ((self.γ ** self.js * self.p_M / self.R - 1) / \
            self.R ** (-self.js)).sum()
        self.c_I = self.w_vals[:, np.newaxis] - \
            self.k_tilde_vals[np.newaxis, :]
        self.c_W = self.c_I[:, :, np.newaxis] + \
            self.ζ_vals[np.newaxis, np.newaxis, :] * self.wage
        self.uc_I = self.u(np.maximum(self.c_I, 1e-50))
        self.uc_W = self.u(np.maximum(self.c_W, 1e-50))
        self.next_w = (1 + self.r) * self.δ_vals[:, np.newaxis, np.newaxis] * \
            self.k_tilde_vals[np.newaxis, :, np.newaxis] + \
            (self.R - (1 + self.r) * self.δ_vals[:, np.newaxis, np.newaxis]) * \
            self.b_vals[np.newaxis, np.newaxis, :]
        self.next_w_star = self.next_w + self.Γ_star

        self.V_init = -10 * np.ones((self.w_vals.size, self.ζ_vals.size,
                                     self.ι_vals.size))


def test_high_Γ_star():
    w_size = 2 ** 9

    params = TestSolveDP(ζ_vals=np.array([0.5, 5.]), M=0.7)

    V_bar_star, π_star = solve_dp_wrapper(params)

    assert not -1 in π_star
    for i in range(3):
        assert np.all(np.diff(V_bar_star, axis=i) >= 0)

    assert π_star[:, 0, 1, 2].sum() >= π_star[:, 0, 0, 2].sum()

    # When Γ_star and π are sufficiently high, it is always optimal to innovate
    assert (V_bar_star[:, 0, 1] == V_bar_star[:, 1, 1]).all()


def test_low_Γ_star():
    params = TestSolveDP(ζ_vals=np.array([3., 6.]), M=3)

    V_bar_star, π_star = solve_dp_wrapper(params)

    assert not -1 in π_star
    for i in range(3):
        assert np.all(np.diff(V_bar_star, axis=i) >= 0)

    assert π_star[:, 0, 1, 2].sum() >= π_star[:, 0, 0, 2].sum()

    # When Γ_star is lower than the minimum wage, it is never optimal to
    # innovate
    assert (V_bar_star[:, 0, 1] == V_bar_star[:, 0, 0]).all()
    assert (V_bar_star[:, 1, 1] == V_bar_star[:, 1, 0]).all()
    assert (π_star[:, 0, 1] == π_star[:, 0, 0]).all()
    assert (π_star[:, 1, 1] == π_star[:, 1, 0]).all()


def test_relax_borrowing_constraint():
    params0 = TestSolveDP()

    params1 = TestSolveDP(b_min=-20., k_tilde_min=-20.)

    V_bar_star0, π_star0 = solve_dp_wrapper(params0)
    V_bar_star1, π_star1 = solve_dp_wrapper(params1)


    assert not -1 in π_star0
    assert not -1 in π_star1

    for i in range(3):
        assert np.all(np.diff(V_bar_star0, axis=i) >= 0)
        assert np.all(np.diff(V_bar_star1, axis=i) >= 0)

    assert π_star0[:, 0, 1, 2].sum() >= π_star0[:, 0, 0, 2].sum()
    assert π_star1[:, 0, 1, 2].sum() >= π_star1[:, 0, 0, 2].sum()

    # Relaxing the borrowing constraint should not decrease the optimal value
    for i in range(2):
        for j in range(2):
            assert (V_bar_star1[:, i, j] >= V_bar_star0[:, i, j]).all()

