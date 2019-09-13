"""
Tests for 'knightian_model.py"

"""

import numpy as np
from knightian_model.discrete import Household, Firm, KnightianInnovationModel
import pytest


def test_P_δ_invalid():
    P_δ_neg = np.array([-0.5, 0.5])
    P_δ_1 = np.array([1.5, 0.5])
    P_δ_sum = np.array([0.25] * 5)

    for P_δ_invalid in [P_δ_neg, P_δ_1, P_δ_sum]:
        with pytest.raises(ValueError) as e_info:
            Household(P_δ=P_δ_invalid)


def test_shapes_invalid():
    δ_vals = np.array([1., 1.1, 1.2])
    P_δ = np.array([1 / 2, 1 / 2])

    with pytest.raises(ValueError) as e_info:
        Household(δ_vals=δ_vals, P_δ=P_δ)

    ζ_vals = np.array([0.5, 1.5, 2.5])
    P_ζ0 = np.array([1 / 2, 1 / 2])

    P_ζ1 = np.array([[1 / 3, 1 / 3, 1 / 3],
                     [1 / 3, 1 / 3, 1 / 3]])

    for P_ζ_invalid in [P_ζ0, P_ζ1]:
        with pytest.raises(ValueError) as e_info:
            Household(ζ_vals=ζ_vals, P_ζ=P_ζ1)


def test_α_invalid():
    α = 1.1

    with pytest.raises(ValueError) as e_info:
        Household(α=α)


def test_β_invalid():
    β = 1.1

    with pytest.raises(ValueError) as e_info:
        Household(β=β)


def test_π_invalid():
    π = 1.1

    with pytest.raises(ValueError) as e_info:
        Household(π=π)


def test_invalid_aggregate():
    invalid_aggregate = -0.5

    hh = Household()
    firms = Firm()

    with pytest.raises(ValueError) as e_info:
        KnightianInnovationModel(hh, firms, K=invalid_aggregate)

    with pytest.raises(ValueError) as e_info:
        KnightianInnovationModel(hh, firms, L=invalid_aggregate)

    with pytest.raises(ValueError) as e_info:
        KnightianInnovationModel(hh, firms, M=invalid_aggregate)
