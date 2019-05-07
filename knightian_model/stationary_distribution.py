import numpy as np
from interpolation import interp
from numba import njit, prange

@njit(parallel=True)
def MC(popu, π_star, w_vals, ζ_vals, δ_vals, Γ_star, P_ζ, P_δ, μ, π, r, R,
       seed=1234, maxiter=1000, tol=1e-5, verbose=True):
    """
    Monte Carlo simulation.
    """

    N = popu.shape[0] // 2

    np.random.seed(seed)

    # is there any gain by taking draws together outside the loop?
    ι_draw_rvs = np.random.random((maxiter, 2*N))
    δ_draw_rvs = np.random.random((maxiter, 2*N))
    π_draw_rvs = np.random.random((maxiter, 2*N))
    ζ_draw_rvs = np.random.random((maxiter, 2*N))

    for i in range(maxiter):
        popu_old = np.copy(popu)
        for j in prange(2*N):
            ι = 0 if ι_draw_rvs[i, j] < μ else 1

            # update w
            w = popu[j, 0]
            ζ_i = int(popu[j, 1])
            ζ = ζ_vals[ζ_i]

            k_tilde = interp(w_vals, π_star[ι, ζ_i, :, 1], w)
            b = interp(w_vals, π_star[ι, ζ_i, :, 2], w)

            δ_i = 0 if δ_draw_rvs[i, j] < P_δ[0] else 1
            δ = δ_vals[δ_i]

            next_w = (1 + r) * δ * k_tilde +(R - (1+r)  * δ) * b

            # the invention decision
            p_invention = interp(w_vals, π_star[ι, ζ_i, :, 0], w)
            if p_invention > 0.5:
            	if π_draw_rvs[i, j] < π:
                	next_w += Γ_star

            popu[j, 0] = next_w

            # update ζ
            # now it only works with two types of ζ
            if ζ_draw_rvs[i, j] > P_ζ[ζ_i, ζ_i]:
                popu[j, 1] = 1 - ζ_i
