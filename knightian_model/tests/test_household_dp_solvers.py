"""
Tests for household_dp_solvers.py

"""

import numpy as np
from ..household_dp_solvers import (_check_approx_fixed_point, update_V2_bf,
                                    update_V1_bf, solve_dp_vi)
from ..function_factories import (utility_function_factory,
                                  production_function_factory)
from ..helper_functions import (create_uc_grid, initialize_values_and_policies,
                                create_next_w, create_P)


# TODO: Add tests for `update_V1_bf`, `update_V2_bf`

def test__check_approx_fixed_point():
    "Tests the function `_check_approx_fixed_point`"

    n = 20
    tol0 = 1e-8
    tol1 = 1e-6
    V1 = np.zeros(n)
    V1_star = np.array([1e-7] * n)
    verbose = False

    assert not _check_approx_fixed_point(V1_star, V1, tol0, verbose)
    assert _check_approx_fixed_point(V1_star, V1, tol1, verbose)

    assert not _check_approx_fixed_point(-V1_star, V1, tol0, verbose)
    assert _check_approx_fixed_point(-V1_star, V1, tol1, verbose)


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
