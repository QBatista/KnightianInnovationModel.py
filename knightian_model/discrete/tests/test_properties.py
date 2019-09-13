"""
Tests for solution properties of the model.

"""

import numpy as np
from knightian_model.discrete import Household, Firm, KnightianInnovationModel


# TODO(QBatista): use pytest fixtures

# The following tests do not constitue proofs. They are checks that ensure the
# quality of the solution.

def check_weakly_increasing(model, tol_nonnegative=1e-8):
    for V in [model.V1_star, model.V2_star]:
        for i in range(V.ndim):
            assert (np.diff(V, axis=i) >= -tol_nonnegative).all()


def check_higher_V(model0, model1):
    assert (model1.V1_star >= model0.V1_star).all()
    assert (model1.V2_star >= model0.V2_star).all()


def do_test(baseline_model, modified_model):
        modified_model.solve_household_DP_problem()

        check_weakly_increasing(modified_model)
        check_higher_V(baseline_model, modified_model)


class TestProperties:
    """Test the solution properties of KnightianInnovationModel"""
    # TODO(QBatista): Speed up the tests with appropriate initial guesses for V_star

    def setup(self):
        self.baseline_hh = Household()
        self.baseline_firms = Firm()
        self.baseline_model = KnightianInnovationModel(self.baseline_hh,
                                                       self.baseline_firms)
        self.baseline_model.solve_household_DP_problem()
        self.ϵ = 1.05

        check_weakly_increasing(self.baseline_model)

    def test_higher_A(self):
        modified_firms = Firm(A=self.baseline_firms.A * self.ϵ)
        modified_model = KnightianInnovationModel(Household(),
                                                  modified_firms)

        do_test(self.baseline_model, modified_model)

    def test_higher_γ(self):
        modified_model = KnightianInnovationModel(Household(),
                                                  Firm(),
                                                  γ=self.baseline_model.γ * self.ϵ)

        do_test(self.baseline_model, modified_model)

    def test_higher_π(self):
        modified_hh = Household(π=self.baseline_hh.π * self.ϵ)
        modified_model = KnightianInnovationModel(modified_hh, Firm())

        do_test(self.baseline_model, modified_model)

    # def test_higher_μ(self):
    #     modified_hh = Household(μ=self.baseline_hh.μ * self.ϵ)
    #     modified_model = KnightianInnovationModel(modified_hh, Firm())
    #
    #     do_test(self.baseline_model, modified_model)

    def test_higher_δ(self):
        modified_hh = Household(δ_vals=self.baseline_hh.δ_vals * self.ϵ)
        modified_model = KnightianInnovationModel(modified_hh, Firm())

        do_test(self.baseline_model, modified_model)

    def test_lower_b_min(self):
        current_b_min = self.baseline_hh.b_vals.min()
        modified_b_min = current_b_min - 1.
        modified_b_vals = np.insert(self.baseline_hh.b_vals, 0, modified_b_min)
        modified_hh = Household(b_vals=modified_b_vals)
        modified_model = KnightianInnovationModel(modified_hh, Firm())

        do_test(self.baseline_model, modified_model)

    def test_higher_b_max(self):
        current_b_max = self.baseline_hh.b_vals.max()
        modified_b_max = current_b_max - 1.
        modified_b_vals = np.insert(self.baseline_hh.b_vals, -1, modified_b_max)
        modified_hh = Household(b_vals=modified_b_vals)
        modified_model = KnightianInnovationModel(modified_hh, Firm())

        do_test(self.baseline_model, modified_model)

    def test_lower_risk_free_rate(self):
        modified_model = KnightianInnovationModel(Household(),
                                                  Firm(),
                                                  R=self.baseline_model.R / self.ϵ)

        do_test(self.baseline_model, modified_model)
