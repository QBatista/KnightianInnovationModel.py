"""
A module with methods for solving an household's dynamic problem.

"""

import numpy as np
from numba import njit, prange
from interpolation import interp
from collections import namedtuple


# TODO: Add documentation, get policy function, modify convergence check to
# also check V2_star, make `verbose` prettier

results = namedtuple('results', 'success num_iter')

# Note: V1 is V_bar, V2 is the intermediate value function

# Value Iteration with Reiter's trick


@njit
def solve_dp_vi(V1_star, V1, V2_star, V2, states_vals, δ_vals, π, β, method,
                method_args, tol=1e-8, maxiter=1000, verbose=True):
    """
    .. highlight:: none

    Solves the household's DP problem using a value iteration algorithm.

    This function is JIT-compiled in `nopython` mode using Numba.

    Parameters
    ----------
    V1_star : ndarray(float, ndim=3)
        Initial guess of the first value function to be modified inplace with
        an approximate fixed point.

    V1 : ndarray(float, ndim=3)
        Array used to store the previous guess of the first value function.

    V2_star : ndarray(float, ndim=3)
        Initial guess of the second value function to be modified inplace with
        an approximate fixed point.

    V1 : ndarray(float, ndim=3)
        Array used to store the previous guess of the second value function.

    states_vals : tuple
        Tuple of ndarray containing the approximation nodes for the state
        variables in the following order: w_vals, ζ_vals, ι_vals, k_tilde_vals.

    δ_vals : ndarray(float, ndim=1)
        Array containing the possible values that δ can take.

    π : scalar(float)
        Probability of an invention being successful.

    β : scalar(float)
        Discount factor.

    method : scalar(int)
        Integer representing the method to be used for solving tmaximization
        problems.
        ::

            0 : Brute force.

    method_args : tuple
        Tuple containing additional arguments required by the choice of
        `method`.
        ::

            0 : P, uc, b_vals, k_tilde_av, b_av, next_w_star, next_w

    tol : scalar(float), optional(default=1e-8)
        Tolerance to be used for determining whether an approximate fixed point
        has been found.

    maxiter : scalar(int), optional(default=1000)
        Maximum number of iterations.

    verbose : bool, optional(default=True)
        If True, prints the sup norm between the current and previous guess
        of the value function at each iteration.

    Returns
    ----------
    results : namedtuple
        A namedtuple containing the following items:
        ::

            "success" : 1 if an approximate fixed point was found; 0 otherwise.
            "num_iter" : Number of iterations performed.

    """
    w_vals, ζ_vals, ι_vals, k_tilde_vals = states_vals

    success = 0

    if method == 0:  # Brute force
        P, uc, b_vals, k_tilde_av, b_av, next_w_star, next_w = \
            method_args

        # Iterate until convergence
        for num_iter in range(maxiter):
            iterate_bf(V1_star, V1, V2, w_vals, ζ_vals, ι_vals, k_tilde_vals,
                       δ_vals, π, β, P, uc, b_vals, k_tilde_av, b_av,
                       next_w_star, next_w, tol, verbose)

            fp1 = _check_approx_fixed_point(V1_star, V1, tol, verbose)

            V1[:] = V1_star

            if fp1:  # Found approximate fixed point
                success = 1
                break

    out = results(success, num_iter)

    return out


@njit(parallel=True)
def iterate_bf(V1_star, V1, V2, w_vals, ζ_vals, ι_vals, k_tilde_vals, δ_vals,
               π, β, P, uc, b_vals, k_tilde_av, b_av, next_w_star, next_w,
               tol, verbose):
    """

    """

    update_V2_bf(ζ_vals, k_tilde_vals, V2, δ_vals, P, w_vals, V1_star,  π,
                 b_vals, b_av, next_w_star, next_w, ι_vals)
    update_V1_bf(ι_vals, ζ_vals, w_vals, V1_star, V2, β, k_tilde_vals, uc,
                 k_tilde_av)


@njit(parallel=True)
def update_V2_bf(ζ_vals, k_tilde_vals, V2, δ_vals, P, w_vals, V1_star,  π,
                 b_vals, b_av, next_w_star, next_w, ι_vals):
    """

    """
    b_av[:] = 0.
    for ζ_i in prange(ζ_vals.size):
        for k_tilde_i in prange(k_tilde_vals.size):
            for b_i in prange(b_vals.size):
                for δ_i in prange(δ_vals.size):
                    for next_ζ_i in prange(ζ_vals.size):
                        for ι in prange(ι_vals.size):
                            b_av[ζ_i, k_tilde_i, b_i, 0] += \
                                P[δ_i, ζ_i, next_ζ_i, ι] * \
                                interp(w_vals, V1_star[ι, next_ζ_i, :],
                                       next_w[δ_i, k_tilde_i, b_i])

                            b_av[ζ_i, k_tilde_i, b_i, 1] += \
                                P[δ_i, ζ_i, next_ζ_i, ι] * \
                                interp(w_vals, V1_star[ι, next_ζ_i, :],
                                       next_w_star[δ_i, k_tilde_i, b_i])


                b_av[ζ_i, k_tilde_i, b_i, 1] += \
                    (π - 1) * (b_av[ζ_i, k_tilde_i, b_i, 1] -
                               b_av[ζ_i, k_tilde_i, b_i, 0])

            V2[ζ_i, k_tilde_i, 0] = b_av[ζ_i, k_tilde_i, :, 0].max()
            V2[ζ_i, k_tilde_i, 1] = b_av[ζ_i, k_tilde_i, :, 1].max()


@njit(parallel=True)
def update_V1_bf(ι_vals, ζ_vals, w_vals, V1_star, V2, β, k_tilde_vals, uc,
                 k_tilde_av):
    """

    """

    for ι in prange(ι_vals.size):
        for ζ_i in prange(ζ_vals.size):
            for w_i in prange(w_vals.size):
                for k_tilde_i in prange(k_tilde_vals.size):
                    k_tilde_av[ι, ζ_i, w_i, k_tilde_i] = \
                        uc[ι, ζ_i, w_i, k_tilde_i] + β * V2[ζ_i, k_tilde_i, ι]

    for ζ_i in prange(ζ_vals.size):
        for w_i in prange(w_vals.size):
            V1_star[0, ζ_i, w_i] = k_tilde_av[0, ζ_i, w_i, :].max()
            V1_star[1, ζ_i, w_i] = max(V1_star[0, ζ_i, w_i],
                                       k_tilde_av[1, ζ_i, w_i, :].max())


@njit
def _check_approx_fixed_point(V_current, V_previous, tol, verbose):
    """
    Checks whether the value iteration algorithm has reached an approximate
    fixed point using the sup norm.

    Parameters
    ----------
    V_current : ndarray(float)
        Most recent approximation of the value function.

    V_previous : ndarray(float)
        Approximation of the value function from the previous iteration of
        the algorithm.

    tol : scalar(float)
        Tolerance to be used for determining whether an approximate fixed point
        has been found.

    verbose : bool
        If True, the sup norm of `V_current` and `V_previous` is printed.

    Returns
    -------
    fp : bool
        Whether `V_current` is an approximate fixed point.

    """

    # Compute the sup norm between `V_current` and `V_previous`
    sup_norm = np.max(np.abs(V_current - V_previous))

    if verbose:
        print(sup_norm)

    # Algorithm termination condition
    fp = sup_norm <= tol

    return fp
