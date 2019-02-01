"""
Implements useful wrappers for testing.

"""

from ..household_dp_solvers import solve_dp_vi


def solve_dp_wrapper(params):
    """
    Call `solve_dp_vi` using the parameters in `params`.

    """
    return solve_dp_vi(params.V_init, params.ι_vals, params.ζ_vals,
                       params.k_tilde_vals, params.b_vals, params.δ_vals,
                       params.next_w, params.next_w_star, params.P_δ,
                       params.P_ζ, params.μ,
                    params.π, params.w_vals, params.wage,
                    params.uc_I, params.uc_W, params.β)
