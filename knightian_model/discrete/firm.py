"""
A module for representing firms in the Knightian innovation model.

"""

from knightian_model.utilities import production_function_factory


class Firm():
    """
    A class for representing firms in the Knightian model of innovation.

    Parameters
    ----------
    A : scalar(float), optional(default=3.)
        Scale of production parameter.

    σ_1 : scalar(float), optional(default=0.3)
        Capital share of production.

    σ_2 : scalar(float), optional(default=0.3)
        Labor share of production.

    Attributes
    ----------
    A, σ_1, σ_2 : See Parameters

    F : callable
        A JIT-compiled utility function that takes capital `K`, labor `L` and
        intermediate goods `M` as inputs and returns the corresponding output.

    F_K : callable
        The JIT-compiled derivative function of `F` with respect to `K`.

    F_L : callable
        The JIT-compiled derivative function of `F` with respect to `L`.

    F_M : callable
        The JIT-compiled derivative function of `F` with respect to `M`.

    """

    def __init__(self, A=3., σ_1=0.3, σ_2=0.3):
        # The following also checks that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(A, σ_1, σ_2)

        self._A = A
        self._σ_1 = σ_1
        self._σ_2 = σ_2

    def __repr__(self):
        out = \
        """
        Firm parameters
        --------------------------------
        Scale of production parameter A = %s
        Capital share parameter σ_1 = %s
        Labor share parameter σ_2 = %s

        """ % (self.A, self.σ_1, self.σ_2)

        return out

    @property
    def A(self):
        "Get the current value of A."

        return self._A

    @A.setter
    def A(self, value):
        # Update production functions and check that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(value, self.σ_1, self.σ_2)
        self._A = A

    @property
    def σ_1(self):
        "Get the current value of σ_1."

        return self._σ_1

    @σ_1.setter
    def σ_1(self, value):
        # Update production functions and check that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(self.A, value, self.σ_2)
        self._σ_1 = value

    @property
    def σ_2(self):
        "Get the current value of σ_2."

        return self._σ_2

    @σ_2.setter
    def σ_2(self, value):
        # Update production functions and check that the parameters are valid
        self._F, self._F_K, self._F_L, self._F_M = \
            production_function_factory(self.A, self.σ_1, value)
        self._σ_1 = value

    @property
    def F(self):
        "Get the current production function F."

        return self._F

    @property
    def F_K(self):
        """
        Get the current derivative function of the production function F with
        respect to K.
        """

        return self._F_K

    @property
    def F_L(self):
        """
        Get the current derivative function of the production function F with
        respect to L.
        """

        return self._F_L

    @property
    def F_M(self):
        """
        Get the current derivative function of the production function F with
        respect to M.
        """

        return self._F_M
