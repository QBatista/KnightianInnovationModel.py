"""
A module for generating parametrized utility and production functions.

"""

from numba import njit


def utility_function_factory(ν):
    """
    Create a CRRA utility function for consumtpion with degree of relative
    risk aversion `ν`.

    Parameters
    ----------
    ν : scalar(float)
        Degree of relative risk aversion

    Returns
    ----------
    utility_function : callable
        A JIT-compiled utility function that takes consumption `c` as an input
        and returns the corresponding CRRA utility.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Isoelastic_utility

    """

    if not ν > 1:
        raise ValueError('ν must be greater than 1.')

    @njit
    def utility_function(c):
        return c ** (1 - ν) / (1 - ν)

    return utility_function


def production_function_factory(A, σ_1, σ_2):
    """
    Create a Cobb-Douglas production function with capital, labor and
    intermediary good inputs and its corresponding derivatives.

    Parameters
    ----------
    A : scalar(float)
        Scale of production parameter.

    σ_1 : scalar(float)
        Capital share of production.

    σ_2 : scalar(float)
        Labor share of production.

    Returns
    ----------
    F : callable
        A JIT-compiled utility function that takes capital `K`, labor `L` and
        intermediate goods `M` as inputs and returns the corresponding output.

    F_K : callable
        The JIT-compiled derivative function of `F` with respect to `K`.

    F_L : callable
        The JIT-compiled derivative function of `F` with respect to `L`.

    F_M : callable
        The JIT-compiled derivative function of `F` with respect to `M`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function

    """

    if not A > 0:
        raise ValueError("A must be strictly greater than 0.")

    if not (σ_1 > 0 and σ_1 < 1):
        raise ValueError('σ_1 must be in (0, 1).')

    if not (σ_2 > 0 and σ_2 < 1):
        raise ValueError('σ_2 must be in (0, 1).')

    @njit
    def F(K, L, M):
        return A * K ** σ_1 * L ** σ_2 * M ** (1 - σ_1 - σ_2)

    @njit
    def F_K(K, L, M):
        return A * σ_1 * K ** (σ_1 - 1) * L ** σ_2 * M ** (1 - σ_1 - σ_2)

    @njit
    def F_L(K, L, M):
        return A * σ_2 * K ** σ_1 * L ** (σ_2 - 1) * M ** (1 - σ_1 - σ_2)

    @njit
    def F_M(K, L, M):
        return A * (1 - σ_1 - σ_2) * K ** σ_1 * L ** σ_2 * M ** (-σ_1 - σ_2)

    return F, F_K, F_L, F_M
