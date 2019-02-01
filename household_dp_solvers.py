"""
Implements methods for solving an household's dynamic problem.

"""

import numpy as np
from numba import njit, prange
from interpolation import interp
import torch
import torch.nn as nn


# Value Iteration with Reiter's trick

@njit()
def solve_dp_vi(V_init, ι_vals, ζ_vals, k_tilde_vals, b_vals, δ_vals,
                next_w, next_w_star, P_δ, P_ζ, μ, π, w_vals, wage,
                uc_I, uc_W, β, tol=1e-8, maxiter=1000, verbose=True):
    """
    Solve a household's dynamic programming problem using a value iteration
    algorithm using Michael Reiter's trick.

    """
    # Initialize value functions
    V_bar = V_init.copy()
    V_bar_interm = np.zeros((k_tilde_vals.size, ζ_vals.size, 2))
    V_bar_new = V_bar.copy()

    # Initialize policy functions
    π_star = -np.ones((w_vals.size, ζ_vals.size, ι_vals.size, 3),
                      dtype=np.int32)

    π_star_interm = np.zeros((k_tilde_vals.size, ζ_vals.size, ι_vals.size),
                              dtype=np.int32)
    π_star_new = π_star.copy()

    # No option to innovate
    π_star_new[:, :, 0, 2] = 0

    # Initialize state action values
    b_action_values = np.zeros((ζ_vals.size, k_tilde_vals.size, b_vals.size,
                                2))
    k_tilde_action_values = np.zeros((w_vals.size, ζ_vals.size,
                                      k_tilde_vals.size, 2))

    bellman_error = np.inf

    for i in range(maxiter):
        # Compute value function update
        update_intermediate_V(b_action_values, V_bar_interm, V_bar_new,
                              π_star_interm, ζ_vals, k_tilde_vals, b_vals,
                              δ_vals, next_w, next_w_star, P_δ, P_ζ, μ, π,
                              w_vals)

        update_V_bar_new(k_tilde_action_values, V_bar_interm, V_bar_new,
                         π_star_new, π_star_interm, β, w_vals, ζ_vals,
                         k_tilde_vals, wage, uc_I, uc_W)

        # Compute error metrics
        bellman_error = np.max(np.abs(V_bar_new - V_bar))
        policy_stability_metric = np.sum(π_star_new != π_star)

        # Update for next iteration
        V_bar[:] = V_bar_new
        π_star[:] = π_star_new

        b_action_values[:] = 0.

        if verbose:
            print(bellman_error, policy_stability_metric)

        # Termination condition
        if bellman_error <= tol:
            break

    return V_bar_new, π_star


@njit(parallel=True)
def update_intermediate_V(b_action_values, V_bar_interm, V_bar_new,
                          π_star_interm, ζ_vals, k_tilde_vals, b_vals, δ_vals,
                          next_w, next_w_star, P_δ, P_ζ, μ, π, w_vals):
    """
    Update V_bar_interm once.

    """
    for k_tilde_i in prange(k_tilde_vals.size):
        for ζ_i in prange(ζ_vals.size):
            for b_i in prange(b_vals.size):
                for δ_i in prange(δ_vals.size):
                    for next_ζ_i in prange(ζ_vals.size):
                        # Continuation value
                        b_action_values[ζ_i, k_tilde_i, b_i, 0] += P_δ[δ_i] * \
                            P_ζ[ζ_i, next_ζ_i] * (
                            (1 - μ) * interp(w_vals, V_bar_new[:, next_ζ_i, 0],
                                       next_w[δ_i, k_tilde_i, b_i]) +
                            μ * interp(w_vals, V_bar_new[:, next_ζ_i, 1],
                                       next_w[δ_i, k_tilde_i, b_i]))

                        b_action_values[ζ_i, k_tilde_i, b_i, 1] += P_δ[δ_i] * \
                            P_ζ[ζ_i, next_ζ_i] * (
                            (1 - μ) * interp(w_vals, V_bar_new[:, next_ζ_i, 0],
                                       next_w_star[δ_i, k_tilde_i, b_i]) +
                            μ * interp(w_vals, V_bar_new[:, next_ζ_i, 1],
                                       next_w_star[δ_i, k_tilde_i, b_i]))

                b_action_values[ζ_i, k_tilde_i, b_i, 1] = \
                    π * b_action_values[ζ_i, k_tilde_i, b_i, 1] + \
                    (1 - π) * b_action_values[ζ_i, k_tilde_i, b_i, 0]

            # Update intermediate value and policy functions
            π_star_interm[k_tilde_i, ζ_i, 0] = \
                b_action_values[ζ_i, k_tilde_i, :, 0].argmax()
            π_star_interm[k_tilde_i, ζ_i, 1] = \
                b_action_values[ζ_i, k_tilde_i, :, 1].argmax()

            V_bar_interm[k_tilde_i, ζ_i, 0] = \
                b_action_values[ζ_i, k_tilde_i,
                                π_star_interm[k_tilde_i, ζ_i, 0], 0]
            V_bar_interm[k_tilde_i, ζ_i, 1] = \
                b_action_values[ζ_i, k_tilde_i,
                                π_star_interm[k_tilde_i, ζ_i, 1], 1]



@njit(parallel=True)
def update_V_bar_new(k_tilde_action_values, V_bar_interm, V_bar_new,
                     π_star_new, π_star_interm, β, w_vals, ζ_vals,
                     k_tilde_vals, wage, uc_I, uc_W):
    """
    Update V_bar_new once.

    """
    for w_i in prange(w_vals.size):
        for ζ_i in prange(ζ_vals.size):
            for k_tilde_i in prange(k_tilde_vals.size):
                # Work
                k_tilde_action_values[w_i, ζ_i, k_tilde_i, 0] = \
                    uc_W[w_i, k_tilde_i, ζ_i] + \
                    β * V_bar_interm[k_tilde_i, ζ_i, 0]

                # Innovate
                k_tilde_action_values[w_i, ζ_i, k_tilde_i, 1] = \
                    uc_I[w_i, k_tilde_i] + β * V_bar_interm[k_tilde_i, ζ_i, 1]

            # Optimal k_tilde for V_W
            π_star_new[w_i, ζ_i, 0, 0] = \
            k_tilde_action_values[w_i, ζ_i, :, 0].argmax()

            # Optimal b
            π_star_new[w_i, ζ_i, 0, 1] = \
            π_star_interm[π_star_new[w_i, ζ_i, 0, 0], ζ_i, 0]

            # Optimal V_bar
            V_bar_new[w_i, ζ_i, 0] = \
            k_tilde_action_values[w_i, ζ_i, π_star_new[w_i, ζ_i, 0, 0], 0]

            # Optimal k_tilde for V_I
            π_star_new[w_i, ζ_i, 1, 0] = \
            k_tilde_action_values[w_i, ζ_i, :, 1].argmax()

            # Optimal decision to innovate conditional on opportunity
            π_star_new[w_i, ζ_i, 1, 2] = \
            (k_tilde_action_values[w_i, ζ_i, π_star_new[w_i, ζ_i, 1, 0], 1] >
             k_tilde_action_values[w_i, ζ_i, π_star_new[w_i, ζ_i, 0, 0], 0])

            if π_star_new[w_i, ζ_i, 1, 2]:
                # No need to update optimal k_tilde
                π_star_new[w_i, ζ_i, 1, 1] = \
                π_star_interm[π_star_new[w_i, ζ_i, 1, 0], ζ_i, 1]
                V_bar_new[w_i, ζ_i, 1] = \
                k_tilde_action_values[w_i, ζ_i, π_star_new[w_i, ζ_i, 1, 0], 1]
            else:
                # Don't innovate
                π_star_new[w_i, ζ_i, 1, 0] = π_star_new[w_i, ζ_i, 0, 0]
                π_star_new[w_i, ζ_i, 1, 1] = \
                π_star_interm[π_star_new[w_i, ζ_i, 0, 0], ζ_i, 0]
                V_bar_new[w_i, ζ_i, 1] = \
                k_tilde_action_values[w_i, ζ_i, π_star_new[w_i, ζ_i, 0, 0], 0]


# Semi-Gradient Descent with a Neural Network

class NeuralNet(nn.Module):
    def __init__(self, input_size, size_hidden_layers, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, size_hidden_layers)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(size_hidden_layers, size_hidden_layers)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(size_hidden_layers, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        return out


def solve_dp_sgd_nn(uc_W, uc_I, P, β, α, ι_vals, δ_vals, ζ_vals, k_tilde_vals,
                    b_vals, states, next_states, next_states_star,
                    size_hidden_layers, device, tol=1e-6):
    """
    Solve a household's dynamic programming problem using a semi-gradient
    descent algorithm with a neural network to approximate the value function.

    """
    nb_states = 3

    V = NeuralNet(nb_states, size_hidden_layers, 1)

    V.to(device)

    criterion = nn.MSELoss()
    optimizer_V = torch.optim.Adam(V.parameters(), lr=α)

    while True:
        # Zero the gradients
        optimizer_V.zero_grad()

        # Compute action values
        Q_W = uc_W + β * (P *
        V(next_states).view(ι_vals.size * δ_vals.size * ζ_vals.size,
                            k_tilde_vals.size * b_vals.size)).sum(0, keepdim=True)
        Q_I = uc_I + β * (P *
        V(next_states_star).view(ι_vals.size * δ_vals.size * ζ_vals.size,
                                 k_tilde_vals.size * b_vals.size)).sum(0, keepdim=True)

        Q_I = π * Q_W + (1 - π) * Q_I

        # Compute gradient descent target
        V_W = torch.max(Q_W, 1, keepdim=True)[0]
        V_I = torch.max(Q_I, 1, keepdim=True)[0]

        target = torch.cat([V_W, torch.max(V_W, V_I)])

        # Evaluate current approximation at the states of interest
        value_estimates = V(states)

        # Compute the update
        loss = criterion(value_estimates, target.detach())

        loss.backward()
        optimizer_V.step()

        print(loss.item())
        if loss.mean().item() < tol:
            break

    return V
