"""
Tests for household_dp_solvers.py

"""

import numpy as np
from .._dp_solvers import (_check_approx_fixed_point, bellman_op_V2_gs,
                           bellman_op_V1_gs, solve_dp_vi)
from knightian_model.utilities import (utility_function_factory,
                                       production_function_factory)
from .._grid import (create_uc_grid, initialize_values_and_policies,
                     create_next_w, create_P)


def test__check_approx_fixed_point():
    "Tests the function `_check_approx_fixed_point`"

    n = 20
    tol0 = 1e-8
    tol1 = 1e-6
    V1 = np.zeros(n)
    V1_star = np.array([1e-7] * n)

    assert not _check_approx_fixed_point(V1_star, V1, tol0)[0]
    assert _check_approx_fixed_point(V1_star, V1, tol1)[0]

    assert not _check_approx_fixed_point(-V1_star, V1, tol0)[0]
    assert _check_approx_fixed_point(-V1_star, V1, tol1)[0]


def test_bellman_op_V1_gs():
    "Tests the function `bellman_op_V1_gs`"

    ι_vals = np.array([0, 1])
    ζ_vals = np.array([0, 1])
    k_tilde_vals = np.array([1., 2., 3.])
    w_vals = np.array([1., 2., 3.])
    β = 0.8

    V1 = np.zeros((2, 2, 3))
    V2 = np.zeros((2, 3, 2))
    V2[:, :, 0] = [[0, 2, 3], [0, 2, 3]]
    V2[:, :, 1] = [[1.5, 3.5, 6], [1.5, 3.5, 6]]

    uc = np.array([[[[0., -np.inf, -np.inf],
                     [1., 0., -np.inf],
                     [2., 1., 0.]],
                    [[0.5, -np.inf, -np.inf],
                     [1.5, 0.5, -np.inf],
                     [2.5, 1.5, 0.5]]],
                   [[[0., -np.inf, -np.inf],
                     [1., 0., -np.inf],
                     [2., 1., 0.]],
                    [[0., -np.inf, -np.inf],
                     [1., 0., -np.inf],
                     [2., 1., 0.]]]])

    k_tilde_av = np.zeros((2, 2, 3, 3))

    bellman_op_V1_gs(ι_vals, ζ_vals, w_vals, V1, V2, β, k_tilde_vals, uc,
                     k_tilde_av)

    k_tilde_av_sol = np.array([[[[0., -np.inf, -np.inf],
                                 [1., 1.6, -np.inf],
                                 [2., 2.6, 2.4]],
                                [[0.5, -np.inf, -np.inf],
                                 [1.5, 2.1, -np.inf],
                                 [2.5, 3.1, 2.9]]],
                               [[[1.2, -np.inf, -np.inf],
                                 [2.2, 2.8, -np.inf],
                                 [3.2, 3.8, 4.8]],
                                [[1.2, -np.inf, -np.inf],
                                 [2.2, 2.8, -np.inf],
                                 [3.2, 3.8, 4.8]]]])

    V1_sol = np.array([[[0., 1.6, 2.6],
                        [0.5, 2.1, 3.1]],
                       [[1.2, 2.8, 4.8],
                        [1.2, 2.8, 4.8]]])

    assert np.allclose(k_tilde_av, k_tilde_av_sol)
    assert np.allclose(V1, V1_sol)


def test_bellman_op_V2_gs():
    "Tests the function `bellman_op_V2_gs`"

    π = μ = 1 / 4
    P_ι = np.array([1 - μ, μ])
    P_δ = np.array([1 / 2, 1 / 2])
    P_ζ = np.array([[1 / 3, 2 / 3], [2 / 3, 1 / 3]])

    P = create_P(P_δ, P_ζ, P_ι)

    ζ_vals = np.array([1, 2])
    ι_vals = np.array([1, 2])
    δ_vals = np.array([1, 2])
    k_tilde_vals = np.array([1, 2, 3])
    b_vals = np.array([0, 1])

    w_vals = np.array([1., 2., 3.])
    next_w = np.array([[[1., 1.],
                        [2., 1.5],
                        [3., 2.5]],
                       [[1., 1.5],
                        [2., 2.5],
                        [3., 3.]]])
    next_w_star = next_w + 1 / 2

    V1 = np.array([[[0., 1., 1.75],
                    [0.5, 1.4, 2.0]],
                   [[0.75, 1.6, 2.3],
                    [1.1, 1.9, 2.5]]])

    V2 = np.zeros((ζ_vals.size, k_tilde_vals.size, ι_vals.size))
    b_av = np.zeros((ζ_vals.size, k_tilde_vals.size, b_vals.size, ι_vals.size))

    bellman_op_V2_gs(ζ_vals, k_tilde_vals, V2, δ_vals, P, w_vals, V1, π,
                     b_vals, b_av, next_w_star, next_w, ι_vals)

    # Computed by hand
    b_av_sol = np.array([[[[0.49583333333333335, 0.6088541666666667],
                           [0.721875, 0.8348958333333334]],
                          [[1.4, 1.4807291666666664],
                           [1.3354166666666667, 1.4322916666666665]],
                          [[2.0458333333333334, 2.0458333333333334],
                           [1.8843749999999997, 1.9247395833333332]]],
                         [[[0.3416666666666667, 0.4583333333333333],
                           [0.575, 0.6916666666666667]],
                          [[1.275, 1.3614583333333332],
                           [1.2145833333333333, 1.3161458333333333]],
                          [[1.9666666666666666, 1.9666666666666666],
                           [1.7937500000000002, 1.8369791666666668]]]])

    V2_sol = np.array([[[0.721875, 0.8348958333333334],
                        [1.4, 1.4807291666666664],
                        [2.0458333333333334, 2.0458333333333334]],
                       [[0.575, 0.6916666666666667],
                        [1.275, 1.3614583333333332],
                        [1.9666666666666666, 1.9666666666666666]]])

    assert np.allclose(b_av, b_av_sol)
    assert np.allclose(V2, V2_sol)


def test_solve_dp_vi():
    """
    Tests if the function `solve_dp_vi` successfully converges to an
    approximate fixed point without checking its output.

    """

    # Parameters
    α = 1.
    A = 1.5
    σ_1 = 0.3
    σ_2 = 0.3
    γ = 0.95
    μ = 0.5
    π = 1
    β = 0.9
    ν = 2
    δ_vals = np.array([0.93, 1])
    P_δ = np.array([0.5, 0.5])

    # State Space
    w_min = 1e-8
    w_max = 10.
    w_size = 2 ** 9
    w_vals = np.linspace(w_min, w_max, w_size)

    ζ_vals = np.array([1., 1.5])
    P_ζ = np.array([[0.5, 0.5],
                    [0.5, 0.5]])

    ι_vals = np.array([0., 1.])
    P_ι = np.array([1 - μ, μ])

    b_min = -0.
    b_max = 10.
    b_size = w_size
    b_vals = np.linspace(b_min, b_max, b_size)

    k_tilde_min = b_min
    k_tilde_max = w_max
    k_tilde_size = w_size
    k_tilde_vals = np.linspace(k_tilde_min, k_tilde_max, k_tilde_size)

    u = utility_function_factory(ν)
    F, F_K, F_L, F_M = production_function_factory(A, σ_1, σ_2)

    # Step 1: Guess initial values of K, L, M
    K, L, M = 5, 1, 0.5

    # Step 2: Guess initial value of R
    R = 1.03

    # Step 3: Compute r, w, p_m
    r, wage, p_M = F_K(K, L, M), F_L(K, L, M), F_M(K, L, M)

    # Step 4: Compute j_bar and Γ_star
    j_bar = np.floor(np.log(R / p_M) / np.log(γ))
    js = np.arange(0, j_bar + 1)
    Γ_star = ((γ ** js * p_M / R - 1) / R ** (-js)).sum()

    states_vals = w_vals, ζ_vals, ι_vals, k_tilde_vals

    uc = create_uc_grid(u, states_vals, wage)
    next_w, next_w_star = create_next_w(r, δ_vals, k_tilde_vals, b_vals, R,
                                        Γ_star)

    V1_star, V1_store, V2_star, V2_store, b_av, k_tilde_av, π_star = \
        initialize_values_and_policies(states_vals, b_vals)

    P = create_P(P_δ, P_ζ, P_ι)

    method = 0
    method_args = \
        (P, uc, b_vals, k_tilde_av, b_av, next_w_star, next_w)

    results = solve_dp_vi(V1_star, V1_store, V2_star, V2_store, states_vals,
                          δ_vals, π, β, method, method_args, tol=1e-7)

    assert results.success == 1
