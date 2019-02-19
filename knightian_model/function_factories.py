"""
A module for generating parametrized utility and production functions.

"""

from numba import njit

# TODO: Add documentation


def utility_function_factory(ν):
    if not ν > 1:
        raise ValueError('ν must be greater than 1.')

    def utility_function(c):
        return c ** (1 - ν) / (1 - ν)

    return njit(utility_function)


def production_function_factory(A, σ_1, σ_2):
    if not A > 0:
        raise ValueError("A must be strictly greater than 0.")

    if not (σ_1 > 0 and σ_1 < 1):
        raise ValueError('σ_1 must be in (0, 1).')

    if not (σ_2 > 0 and σ_2 < 1):
        raise ValueError('σ_2 must be in (0, 1).')

    def F(K, L, M):
        return A * K ** σ_1 * L ** σ_2 * M ** (1 - σ_1 - σ_2)

    def F_K(K, L, M):
        return A * σ_1 * K ** (σ_1 - 1) * L ** σ_2 * M ** (1 - σ_1 - σ_2)

    def F_L(K, L, M):
        return A * σ_2 * K ** σ_1 * L ** (σ_2 - 1) * M ** (1 - σ_1 - σ_2)

    def F_M(K, L, M):
        return A * (1 - σ_1 - σ_2) * K ** σ_1 * L ** σ_2 * M ** (-σ_1 - σ_2)

    return njit(F), njit(F_K), njit(F_L), njit(F_M)
