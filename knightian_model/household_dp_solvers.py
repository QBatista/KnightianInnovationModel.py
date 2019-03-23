"""
A module with methods for solving an household's dynamic problem.

"""

import numpy as np
from numba import njit, prange
from interpolation import interp
from collections import namedtuple


# TODO:
# - modify convergence check to check V2_star
# - make tolerance level of error
# - make `verbose` prettier

results = namedtuple('results', 'success num_iter')

# Note: V1 is V_bar, V2 is the intermediate value function

# Value Iteration with Reiter's trick


@njit
def solve_dp_vi(V1_star, V1_store, V2_star, V2_store, states_vals, δ_vals, π,
                β, method, method_args, tol=1e-8, maxiter=1000, verbose=True):
    """
    .. highlight:: none

    Solves the household's DP problem using a value iteration algorithm.

    This function is JIT-compiled in `nopython` mode using Numba.

    Parameters
    ----------
    V1_star : ndarray(float, ndim=3)
        Initial guess of the first value function to be modified inplace with
        an approximate fixed point.

    V1_store : ndarray(float, ndim=3)
        Array used to store the previous guess of the first value function.

    V2_star : ndarray(float, ndim=3)
        Initial guess of the second value function to be modified inplace with
        an approximate fixed point.

    V2_store : ndarray(float, ndim=3)
        Array used to store the previous guess of the second value function.

    states_vals : tuple
        Tuple of ndarray containing the approximation nodes for the state
        variables in the following order: w_vals, ζ_vals, ι_vals, k_tilde_vals.

    δ_vals : ndarray(float, ndim=1)
        Grid for the depreciation rate variable δ.

    π : scalar(float)
        Probability of an invention being successful.

    β : scalar(float)
        Discount factor. Must be strictly less than 1.

    method : scalar(int)
        Integer representing the method to be used for solving tmaximization
        problems.
        ::

            0 : Grid search.

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
            bellman_op_V2_gs(ζ_vals, k_tilde_vals, V2_star, δ_vals, P, w_vals,
                             V1_star, π, b_vals, b_av, next_w_star, next_w,
                             ι_vals)
            bellman_op_V1_gs(ι_vals, ζ_vals, w_vals, V1_star, V2_star, β,
                             k_tilde_vals, uc, k_tilde_av)

            fp1 = _check_approx_fixed_point(V1_star, V1_store, tol, verbose)

            V1_store[:] = V1_star
            V2_store[:] = V2_star

            if fp1:  # Found approximate fixed point
                success = 1
                break

    out = results(success, num_iter)

    return out


@njit(parallel=True)
def bellman_op_V2_gs(ζ_vals, k_tilde_vals, V2, δ_vals, P, w_vals, V1, π,
                     b_vals, b_av, next_w_star, next_w, ι_vals):
    """
    Bellman operator for the second value function using grid search
    maximization.

    Parameters
    ----------
    ζ_vals : ndarray(float, ndim=1)
        Grid for the labor income shock variable ζ.

    k_tilde_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for both the state and
        choice net investment variable k_tilde.

    V2 : ndarray(float, ndim=3)
        Array of shape `(ζ_vals.size, k_tilde_vals.size, ι_vals.size)`
        containing estimates of the second value function.

    δ_vals : ndarray(float, ndim=1)
        Array containing the possible values that δ can take.

    P : ndarray(float, ndim=3)
        Joint probability distribution over the values of δ, ζ and ι.
        Probabilities vary by δ on the first axis, by ζ on the second axis,
        and by ι on the third axis.

    w_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for the wealth state variable
        w.

    V1 : ndarray(float, ndim=3)
        Array of shape `(ι_vals.size, ζ_vals.size, w_vals.size)` to be modified
        inplace with the result of applying the bellman operator.

    π : scalar(float)
        Probability of an invention being successful.

    b_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for the borrowing choice
        variable.

    b_av : ndarray(float, ndim=4)
        Array used to store the action values of the different borrowing
        levels at the approximation nodes `b_vals`.

    next_w_star : ndarray(float, ndim=3)
        Array containing the next period wealth values implied by the
        parameters when an invention is succesful.

    next_w : ndarray(float, ndim=3)
        Array containing the next period wealth values implied by the
        parameters without a succesful invention.

    ι_vals : ndarray(float, ndim=1)
        Grid for the invention opportunity variable ι.

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
                                interp(w_vals, V1[ι, next_ζ_i, :],
                                       next_w[δ_i, k_tilde_i, b_i])

                            b_av[ζ_i, k_tilde_i, b_i, 1] += \
                                P[δ_i, ζ_i, next_ζ_i, ι] * \
                                interp(w_vals, V1[ι, next_ζ_i, :],
                                       next_w_star[δ_i, k_tilde_i, b_i],)

                b_av[ζ_i, k_tilde_i, b_i, 1] += \
                    (π - 1) * (b_av[ζ_i, k_tilde_i, b_i, 1] -
                               b_av[ζ_i, k_tilde_i, b_i, 0])

            V2[ζ_i, k_tilde_i, 0] = b_av[ζ_i, k_tilde_i, :, 0].max()
            V2[ζ_i, k_tilde_i, 1] = b_av[ζ_i, k_tilde_i, :, 1].max()


@njit(parallel=True)
def bellman_op_V1_gs(ι_vals, ζ_vals, w_vals, V1, V2, β, k_tilde_vals, uc,
                     k_tilde_av):
    """
    Bellman operator for the first value function using grid search
    maximization.

    Parameters
    ----------
    ι_vals : ndarray(float, ndim=1)
        Grid for the invention opportunity variable ι.

    ζ_vals : ndarray(float, ndim=1)
        Grid for the labor income shock variable ζ.

    w_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for the wealth state variable
        w.

    V1 : ndarray(float, ndim=3)
        Array of shape `(ι_vals.size, ζ_vals.size, w_vals.size)` to be modified
        inplace with the result of applying the bellman operator.

    V2 : ndarray(float, ndim=3)
        Array of shape `(ζ_vals.size, k_tilde_vals.size, ι_vals.size)`
        containing estimates of the second value function.

    β : scalar(float)
        Discount factor.

    k_tilde_vals : ndarray(float, ndim=1)
        Array containing the approximation nodes for both the state and
        choice net investment variable k_tilde.

    uc : ndarray(float, ndim=4)
        Array of shape `(ι_vals.size, ζ_vals.size, w_vals.size,
        k_tilde_vals.size)` containing utility function evaluations.

    k_tilde_av : ndarray(float, ndim=4)
        Array of shape `(ι_vals.size, ζ_vals.size, w_vals.size,
        k_tilde_vals.size)` used to be modified inplace with the action values
        of the different net investment levels at the approximation nodes
        `k_tilde_vals`.

    """

    for ι in prange(ι_vals.size):
        for ζ_i in prange(ζ_vals.size):
            for w_i in prange(w_vals.size):
                for k_tilde_i in prange(k_tilde_vals.size):
                    k_tilde_av[ι, ζ_i, w_i, k_tilde_i] = \
                        uc[ι, ζ_i, w_i, k_tilde_i] + β * V2[ζ_i, k_tilde_i, ι]

    for ζ_i in prange(ζ_vals.size):
        for w_i in prange(w_vals.size):
            V1[0, ζ_i, w_i] = k_tilde_av[0, ζ_i, w_i, :].max()
            V1[1, ζ_i, w_i] = max(V1[0, ζ_i, w_i],
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
