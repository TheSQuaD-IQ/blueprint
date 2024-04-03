from typing import Any, Union, Callable, Iterable, List, Iterator, Tuple

from jax import Array
from jax import numpy as jnp

from ..base.terms import TimeDependentTerm

Numeric = Union[float, complex]


class Drive:
    """
    Drive A class for representing a drive in the Hamiltonian.
    """

    def __init__(
        self,
        label: str,
        prefactor: Callable | Iterable[Callable],
        operator: Array | Iterable[Array],
    ) -> None:
        if not isinstance(label, str):
            raise ValueError(
                f"The label must be a string, instead got type {type(label)}."
            )
        self._label = label

        if isinstance(operator, Array):
            op_dims = len(operator.shape)
            if op_dims != 2:
                raise ValueError(
                    f"The operator must be a 2D array, instead got a {op_dims}-dimensional array."
                )
            dim, other_dim = operator.shape
            if dim != other_dim:
                raise ValueError(
                    f"The operator must be a square matrix, instead got a ({dim}x{other_dim}) matrix."
                )

            self._op = operator

        elif isinstance(operator, Iterable):
            operators = list(operator)
            dim, _ = operators[0].shape

            for operator in operators:
                op_dims = len(operator.shape)
                if op_dims != 2:
                    raise ValueError(
                        f"The operator must be a 2D array, instead got a {op_dims}-dimensional array."
                    )

                op_dim, other_dim = operator.shape
                if dim != other_dim:
                    raise ValueError(
                        f"The operator must be a square matrix, instead got a ({dim}x{other_dim}) matrix."
                    )

                if op_dim != dim:
                    raise ValueError(
                        f"All operators must have the same dimension, instead got operators with dimensions {op_dim} and {dim}."
                    )

            self._op = operators

        else:
            raise ValueError(
                f"The operator must be a jax.Array or an iterable of jax.Arrays, instead got type {type(operator)}."
            )

        self._dim = dim

        self._prefactor_terms: List[TimeDependentTerm] = []

        if isinstance(prefactor, Callable):
            prefactor_term = TimeDependentTerm(prefactor)
            self._prefactor_terms.append(prefactor_term)
        elif isinstance(prefactor, Iterable):
            prefactors = list(prefactor)
            if isinstance(self._op, list):
                num_ops = len(self._op)
                num_prefactors = len(prefactors)

                if num_ops != num_prefactors:
                    raise ValueError(
                        "The number of prefactors must match the number of operators, "
                        f"instead got {num_prefactors} prefactors and {num_ops} operators."
                    )

            for single_prefactor in prefactors:
                if not isinstance(single_prefactor, Callable):
                    raise ValueError("Each prefactor in prefactors must be a callable.")

                prefactor_term = TimeDependentTerm(single_prefactor)
                self._prefactor_terms.append(prefactor_term)
        else:
            raise ValueError(
                "The prefactor must be a callable, or an iterable of callables."
            )

    @property
    def label(self) -> str:
        """
        label Returns the label of the time-dependent term.

        Returns
        -------
        str
            The label of the time-dependent term.
        """
        return self._label

    @property
    def dim(self) -> int:
        """
        dim Returns the dimension of the time-dependent term.

        Returns
        -------
        int
            The dimension of the time-dependent term.
        """
        return self._dim

    @property
    def free_params(self) -> List[str]:
        """
        free_params Returns the free parameters of the possibly time-dependent drive term.

        Returns
        -------
        List[str]
            The list of free parameters (str) of the possibly time-dependent drive term.
        """
        params = set()
        for prefactor_term in self._prefactor_terms:
            if isinstance(prefactor_term, TimeDependentTerm):
                params.update(prefactor_term.free_params)
        return list(params)

    @property
    def params(self) -> List[str]:
        """
        params Returns the parameters of the possibly time-dependent drive term. This includes both free and fixed parameters.

        Returns
        -------
        List[str]
            The list of parameters (str) of the possibly time-dependent drive term.
        """
        params = set()
        for prefactor_term in self._prefactor_terms:
            if isinstance(prefactor_term, TimeDependentTerm):
                params.update(prefactor_term.params)
        return list(params)

    @property
    def param_vals(self) -> dict[str, Any]:
        """
        param_vals Returns the parameter values of the possibly time-dependent drive.

        Returns
        -------
        dict[str, Any]
            The dictionary of parameter names (str) and corresponding values (Any) of the possibly time-dependent drive.
        """
        params = {}
        for prefactor_term in self._prefactor_terms:
            if isinstance(prefactor_term, TimeDependentTerm):
                params.update(prefactor_term.param_vals)
        return params

    def set_params(self, **params) -> None:
        """
        set_params Sets the parameters of the time-dependent term.

        Parameters
        ----------
        **params
            The parameters of the time-dependent term.
        """
        for term in self._prefactor_terms:
            if isinstance(term, TimeDependentTerm):
                term_params = term.params
                given_params = {
                    param: val for param, val in params.items() if param in term_params
                }
                if given_params:
                    term.set_params(**given_params)

    def eval_prefactors(self, **params) -> Iterator[Numeric]:
        """
        eval_prefactor Evaluates the prefactor of the time-dependent term.

        Returns
        -------
        Numeric
            The evaluated prefactor.
        """
        for term in self._prefactor_terms:
            yield term.eval_prefactor(**params)

    def _get_hamiltonian(self, **params) -> Array:
        """
        _get_hamiltonian Evaluates the prefactor and returns the Hamiltonian of the coupling term.

        Parameters
        ----------
        exclude_prefactor : bool, optional
            Whether to include the prefactor in front of the quantum operators, by default False

        Returns
        -------
        Array
            The coupling term Hamiltonian.

        Raises
        ------
        ValueError
            If the evaluated prefactor is not a number (float or complex).
        """
        prefactors = list(self.eval_prefactors(**params))

        hamiltonian = jnp.zeros((self._dim, self._dim))
        if isinstance(self._op, list):
            for prefactor, op in zip(prefactors, self._op):
                hamiltonian = jnp.add(hamiltonian, prefactor * op)
        else:
            for prefactor in prefactors:
                hamiltonian = jnp.add(hamiltonian, prefactor * self._op)
        return hamiltonian

    def get_hamiltonian(self, **params) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian of the coupling term.

        Parameters
        ----------
        exclude_prefactor : bool, optional
            Whether to include the prefactor in front of the quantum operators, by default False

        Returns
        -------
        Array
            The coupling term Hamiltonian.
        """
        return self._get_hamiltonian(**params)

    def decompose(self) -> Iterator[Tuple[TimeDependentTerm, Array]]:
        """
        decompose Decomposes the Hamiltonian term associated with this drive
        into the operators and corresponding prefactors that express it.

        Yields
        ------
        Iterator[Tuple[TimeDependentTerm, Array]]
            An iterator of tuples containing the tuples of prefactor and the operator of the drive term.
        """
        if isinstance(self._op, list):
            for prefactor, op in zip(self._prefactor_terms, self._op):
                yield prefactor, op
        else:
            for prefactor in self._prefactor_terms:
                yield prefactor, self._op
