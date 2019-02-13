"""
Tests for function_factories.py

"""

import pytest
from ..function_factories import (utility_function_factory,
                                  production_function_factory)


def test_utility_function_factory():
    "Tests the function `utility_function_factory`"

    tol = 1e-8
    ν = 2.
    c = 11 / 24
    sol = -24 / 11

    u = utility_function_factory(ν)
    uc = u(c)

    assert abs(uc - sol) < tol


def test_utility_function_factory_invalid_ν():
    "Tests the function `utility_function_factory` when the input is invalid"

    ν = 1.

    with pytest.raises(ValueError) as e_info:
        utility_function_factory(ν)


def test_production_function_factory():
    "Tests the function `production_function_factory`"

    tol = 1e-8
    A = 1.
    σ_1, σ_2 = 1 / 3, 1 / 3

    F, F_K, F_L, F_M = production_function_factory(A, σ_1, σ_2)

    K = 1 / 3
    L = 2 / 3
    M = 1.

    F_computed = F(K, L, M)
    F_K_computed = F_K(K, L, M)
    F_L_computed = F_L(K, L, M)
    F_M_computed = F_M(K, L, M)

    F_sol = K ** σ_1 * L ** σ_2 * M ** (1 - σ_1 - σ_2)
    F_K_sol = σ_1 * F_sol / K
    F_L_sol = σ_2 * F_sol / L
    F_M_sol = (1 - σ_1 - σ_2) * F_sol / M

    assert abs(F_computed - F_sol) < tol
    assert abs(F_K_computed - F_K_sol) < tol
    assert abs(F_L_computed - F_L_sol) < tol
    assert abs(F_M_computed - F_M_sol) < tol


def test_production_function_factory_invalid_A():
    "Tests the function `production_function_factory` with invalid input"

    A = 0.
    σ_1, σ_2 = 0.3, 0.3

    with pytest.raises(ValueError) as e_info:
        production_function_factory(A, σ_1, σ_2)


def test_production_function_factory_invalid_σ_1():
    "Tests the function `production_function_factory` with invalid input"

    A = 1.
    σ_1, σ_2 = 1., 0.3

    with pytest.raises(ValueError) as e_info:
        production_function_factory(A, σ_1, σ_2)

    σ_1 = 0.

    with pytest.raises(ValueError) as e_info:
        production_function_factory(A, σ_1, σ_2)


def test_production_function_factory_invalid_σ_2():
    "Tests the function `production_function_factory` with invalid input"

    A = 1.
    σ_1, σ_2 = 0.3, 1.

    with pytest.raises(ValueError) as e_info:
        production_function_factory(A, σ_1, σ_2)

    σ_2 = 0.

    with pytest.raises(ValueError) as e_info:
        production_function_factory(A, σ_1, σ_2)
