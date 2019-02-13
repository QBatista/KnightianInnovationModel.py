"""
Tests for helper_functions.py

"""

import numpy as np
from ..helper_functions import create_uc_grid, create_P
from ..function_factories import utility_function_factory


# TODO: Add tests for `create_next_w`, `initialize_values_and_policies`,
# `create_P`.

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
