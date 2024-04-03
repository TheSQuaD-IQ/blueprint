from __future__ import annotations
from typing import Callable, Dict, Any, List, Union
from copy import copy
from warnings import warn

from inspect import signature

Numeric = Union[float, complex]


class TimeDependentTerm:
    """
    A base class for time-dependent (or fixed) prefactors in the
    Hamiltonian.
    """

    def __init__(self, prefactor: Callable) -> None:
        if not isinstance(prefactor, Callable):
            raise ValueError(
                f"The prefactor expected to be a function (callable), instead got {type(prefactor)}"
            )

        self._params = get_func_params(prefactor)
        self._prefactor: Callable = prefactor

        # Set the is_constant flag
        self._is_constant = False

    def __copy__(self) -> TimeDependentTerm:
        term_copy = self.__class__(self._prefactor)
        term_copy.set_params(**self._params)
        return term_copy

    @property
    def free_params(self) -> List[str]:
        """
        free_params Returns the free parameters of the time-dependent term.

        Returns
        -------
        Dict[str, Any]
            The free parameters of the time-dependent term.
        """
        params = [par for par, val in self._params.items() if val is None]
        return params

    @property
    def params(self) -> List[str]:
        """
        params Returns the (both free and fixed) parameters of the time-dependent term.

        Returns
        -------
        List[str]
            The list of parameters of the time-dependent term.
        """
        return list(self._params)

    @property
    def param_vals(self) -> Dict[str, Any]:
        """
        param_vals Returns a dictionary of the term parameters and their respective values.

        Returns
        -------
        Dict[str, Any]
            The parameters of the time-dependent term.
        """
        return self._params

    @property
    def is_constant(self) -> bool:
        """
        is_constant Returns whether the term is constant.

        Returns
        -------
        bool
            True if the term is constant, False otherwise.
        """
        return False

    def set_params(self, **params) -> None:
        """
        set_params Sets the parameters of the time-dependent term.

        Raises
        ------
        KeyError
            If a parameter is not found in the time-dependent term.
        """
        for param, val in params.items():
            if param not in self._params:
                raise KeyError(
                    f"Parameter {param} not in found in the coefficient parameters"
                )
            self._params[param] = val

    def eval_prefactor(self, **params) -> Numeric:
        """
        eval_prefactor Evaluates the prefactor of the time-dependent term.

        Returns
        -------
        float
            The prefactor of the time-dependent term.
        """
        free_params = set(self.free_params)  # Get the free parameters
        # Get the parameters that have default values.
        set_params = set(self.params) - free_params

        if not params:  # No parameters were given
            if free_params:
                raise ValueError(
                    f"The prefactor has undefined parameters {tuple(free_params)}."
                    "These parameters must be defined to evaluate the prefactor."
                )
            return self._prefactor(**self._params)

        merged_params = copy(self._params)

        for param, val in params.items():
            if param in set_params:
                default_val = self._params[param]
                warning_message = f"Over-writing the parameter {param} with a default value {default_val} with a value of: {val}."
                warn(warning_message)
            if param in free_params:
                free_params.remove(param)
            else:
                warn(
                    f"Given parameter {param} does not parameterize the prefactor, but a value of {val} was provided for the evaluation."
                )

            merged_params[param] = val

        return self._prefactor(**merged_params)

    def __call__(self, **params) -> Numeric:
        """
        __call__ Evaluates the prefactor of the time-dependent term.

        Returns
        -------
        Numeric
            The prefactor of the time-dependent term.
        """
        return self.eval_prefactor(**params)


def get_func_params(func: Callable) -> Dict[str, Any]:
    """
    get_func_params Returns the parameters of a given function.

    Parameters
    ----------
    func : Callable
        The function from which to extract the parameters.

    Returns
    -------
    Dict[str, Any]
        The parameters of the function, returned as a dictionary. Each parameter name is listed as a key. If the parameter has a default value, the value is the default value. If the parameter does not have a default value, the value is None.
    """
    sig = signature(func)
    params = {}

    for param in sig.parameters.values():
        if param.default is param.empty:
            params[param.name] = None
        else:
            params[param.name] = param.default
    return params
