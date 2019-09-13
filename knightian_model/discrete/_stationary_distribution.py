
import numpy as np
from interpolation import interp
from numba import njit, prange


@njit(parallel=True)
def MC(popu, π_star, w_vals, ζ_vals, δ_vals, Γ_star, P_ζ_cdfs, P_δ, μ, π, r, R,
       seed=1234, maxiter=1000, tol=1e-5, verbose=True):
    """
    Monte Carlo simulation.
    """

    N = popu.shape[0]

    # is there any gain by taking draws together outside the loop?
    ι_draw_rvs = np.random.random((maxiter, N))
    δ_draw_rvs = np.random.random((maxiter, N))
    π_draw_rvs = np.random.random((maxiter, N))
    ζ_draw_rvs = np.random.random((maxiter, N))

    ζ_i_draw_rvs = np.zeros((maxiter, N), dtype=np.int64)
    _generate_sample_paths(P_ζ_cdfs, popu[:, 1], ζ_draw_rvs, ζ_i_draw_rvs)
    popu[:, 1] = ζ_i_draw_rvs[-1, :]

    for i in range(maxiter):
        popu_old = np.copy(popu)
        for j in prange(N):
            ι = 0 if ι_draw_rvs[i, j] < μ else 1

            # update w
            w = popu[j, 0]
            ζ_i = ζ_i_draw_rvs[i, j]
            ζ = ζ_vals[ζ_i]

            k_tilde = interp(w_vals, π_star[ι, ζ_i, :, 1], w)
            b = interp(w_vals, π_star[ι, ζ_i, :, 2], w)

            δ_i = 0 if δ_draw_rvs[i, j] < P_δ[0] else 1
            δ = δ_vals[δ_i]

            next_w = (1 + r) * δ * k_tilde + (R - (1+r) * δ) * b

            # the invention decision
            p_invention = interp(w_vals, π_star[ι, ζ_i, :, 0], w)
            if p_invention > 0.5:
            	if π_draw_rvs[i, j] < π:
                	next_w += Γ_star

            popu[j, 0] = next_w


@njit
def _generate_sample_paths(P_cdfs, init_states, random_values, out):
    """
    Generate num_reps sample paths of length ts_length, where num_reps =
    out.shape[0] and ts_length = out.shape[1].
    Parameters
    ----------
    P_cdfs : ndarray(float, ndim=2)
        Array containing as rows the CDFs of the state transition.
    init_states : array_like(int, ndim=1)
        Array containing the initial states. Its length must be equal to
        num_reps.
    random_values : ndarray(float, ndim=2)
        Array containing random values from [0, 1). Its shape must be
        equal to (num_reps, ts_length-1)
    out : ndarray(int, ndim=2)
        Array to store the sample paths.
    Notes
    -----
    This routine is jit-complied by Numba.

    Copied from: https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/core.py
    """
    num_reps, ts_length = out.shape

    for i in range(num_reps):
        out[i, 0] = init_states[i]
        for t in range(ts_length-1):
            out[i, t+1] = np.searchsorted(P_cdfs[out[i, t]], random_values[i, t])
