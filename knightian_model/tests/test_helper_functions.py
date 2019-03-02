"""
Tests for helper_functions.py

"""

import numpy as np
from ..helper_functions import (create_uc_grid, create_P, create_next_w,
                                initialize_values_and_policies)
from ..function_factories import utility_function_factory


def test_create_uc_grid():
    "Tests the function `create_uc_grid`"

    ν = 2.
    u = utility_function_factory(ν)

    w_vals = np.array([1., 2.])
    ζ_vals = np.array([1/4, 1/2])
    ι_vals = np.array([0, 1])
    k_tilde_vals = np.array([0., 1., 2.])

    wage = 1.

    min_c = 1e-10

    states_vals = w_vals, ζ_vals, ι_vals, k_tilde_vals

    uc = create_uc_grid(u, states_vals, wage, min_c=min_c)

    uc_sol = -1 / np.array([[[[5/4, 1/4, 1e-10],
                              [9/4., 5/4, 1/4]],
                             [[3/2, 1/2, 1e-10],
                              [5/2., 3/2, 1/2]]],
                            [[[1., 1e-10, 1e-10],
                              [2., 1., 1e-10]],
                             [[1., 1e-10, 1e-10],
                              [2., 1., 1e-10]]]])

    assert (uc == uc_sol).all()


def test_fixed_create_P():
    "Tests the output of the function `create_P` with a fixed P"

    P_ι = np.array([3 / 4, 1 / 4])
    P_δ = np.array([1 / 2, 1 / 2])
    P_ζ = np.array([[1 / 3, 2 / 3], [2 / 3, 1 / 3]])

    P = create_P(P_δ, P_ζ, P_ι)

    P_sol = np.array([[[[1 / 8, 1 / 24],
                        [1 / 4, 1 / 12]],
                       [[1 / 4, 1 / 12],
                        [1 / 8, 1 / 24]]],
                      [[[1 / 8, 1 / 24],
                        [1 / 4, 1 / 12]],
                       [[1 / 4, 1 / 12],
                        [1 / 8, 1 / 24]]]])

    assert np.allclose(P, P_sol)
    assert P[:, 0, :, :].sum() == 1.
    assert P[:, 1, :, :].sum() == 1.


def test_random_create_P():
    """
    Tests the properties of the output of the function `create_P` with a
    random P.
    """

    max_step = 100
    n = 50
    low = 1
    tol = 1e-8

    P_ι = np.random.dirichlet(np.random.randint(low, max_step, size=n))
    P_δ = np.random.dirichlet(np.random.randint(low, max_step, size=n))
    P_ζ = np.random.dirichlet(np.random.randint(low, high=max_step, size=50),
                              size=2)

    P = create_P(P_δ, P_ζ, P_ι)

    assert abs(P[:, 0, :, :].sum() - 1.) < tol
    assert abs(P[:, 1, :, :].sum() - 1.) < tol


def test_create_next_w():
    "Tests the function `create_next_w`"

    r = 0.05
    δ_vals = np.array([0.98, 1.02])
    k_tilde_vals = np.array([1., 2., 3.])
    R = 1.02
    b_vals = np.array([0., 1.])
    Γ_star = 1 / 2

    next_w, next_w_star = create_next_w(r, δ_vals, k_tilde_vals, b_vals, R,
                                        Γ_star)

    next_w_sol = np.array([[[1.029, 1.02],
                            [2.058, 2.049],
                            [3.087, 3.078]],
                           [[1.071, 1.02],
                            [2.142, 2.091],
                            [3.213, 3.162]]])
    next_w_star_sol = next_w_sol + Γ_star

    assert np.allclose(next_w, next_w_sol)
    assert np.allclose(next_w_star, next_w_star_sol)


def test_initialize_values_and_policies():
    """
    Tests the shape of the output of the function
    `initialize_values_and_policies`
    """

    w_vals = np.arange(10)
    ζ_vals = np.arange(20)
    ι_vals = np.arange(30)
    k_tilde_vals = np.arange(40)
    b_vals = np.arange(50)

    states_vals = w_vals, ζ_vals, ι_vals, k_tilde_vals

    V1_star, V1_store, V2_star, V2_store, b_av, k_tilde_av, π_star = \
        initialize_values_and_policies(states_vals, b_vals)

    V1_shape_sol = (30, 20, 10)
    V2_shape_sol = (20, 40, 30)
    b_av_shape_sol = (20, 40, 50, 30)
    k_tilde_av_shape_sol = (30, 20, 10, 40)
    π_star_shape_sol = (30, 20, 10, 3)

    assert V1_star.shape == V1_shape_sol
    assert V1_star.shape == V1_store.shape
    assert V2_star.shape == V2_shape_sol
    assert V2_star.shape == V2_store.shape
    assert b_av.shape == b_av_shape_sol
    assert k_tilde_av.shape == k_tilde_av_shape_sol
    assert π_star.shape == π_star_shape_sol
