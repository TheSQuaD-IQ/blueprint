from typing import Union, Callable, Iterable, List, Iterator, Tuple

from jax import Array
from jax import numpy as jnp

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

            for operator in operators:
                if not isinstance(operator, Array):
                    raise ValueError(
                        f"Each drive operator must be a jax.Array object, instead got type {type(operator)}."
                    )

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
                f"The drive operator must be either a jax.Array object or an Iterable of jax.Array objects, instead got type {type(operator)}."
            )

        self._dim = dim
        self._prefactors: List[Callable] = []

        if isinstance(prefactor, Callable):
            self._prefactors.append(prefactor)

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

            for op_prefactor in prefactors:
                if not isinstance(op_prefactor, Callable):
                    raise ValueError("Each prefactor in prefactors must be a callable.")

                self._prefactors.append(op_prefactor)

        else:
            raise ValueError(
                f"The drive prefactor must be either a Callable object or an Iterable of Callable objects, instead got type {type(prefactor)}."
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
    def prefactors(self) -> List[Callable]:
        """
        prefactors Returns the prefactors of the time-dependent term.

        Returns
        -------
        List[Partial]
            The prefactors of the time-dependent term.
        """
        return self._prefactors

    @property
    def operator(self) -> Array | List[Array]:
        """
        operator Returns the operator of the time-dependent term.

        Returns
        -------
        Array | List[Array]
            The operator of the time-dependent term.
        """
        return self._op

    def eval_prefactors(self, time: float) -> Iterator[Numeric]:
        """
        eval_prefactor Evaluates the prefactor of the time-dependent term.

        Returns
        -------
        Numeric
            The evaluated prefactor.
        """
        for prefactor in self._prefactors:
            prefactor_val = prefactor(time)
            yield prefactor_val

    def _get_hamiltonian(self, time: float) -> Array:
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
        prefactors = list(self.eval_prefactors(time))

        hamiltonian = jnp.zeros((self._dim, self._dim))
        if isinstance(self._op, list):
            for prefactor, op in zip(prefactors, self._op):
                hamiltonian = hamiltonian + prefactor * op
            return hamiltonian

        for prefactor in prefactors:
            hamiltonian = hamiltonian + prefactor * self._op
        return hamiltonian

    def get_hamiltonian(self, time: float) -> Array:
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
        return self._get_hamiltonian(time)

    def decompose(self) -> Iterator[Tuple[Callable, Array]]:
        """
        decompose Decomposes the drive term into prefactors and operators.

        Parameters
        ----------
        finalize : bool, optional
            Whether to finalize the prefactors, by default True

        Yields
        ------
        Iterator[Tuple[Callable, Array]]
            The prefactors and operators of the drive term.

        Raises
        ------
        ValueError
            If a required positional argument is missing.
        """
        if isinstance(self._op, list):
            yield from zip(self._prefactors, self._op)
        else:
            for prefactor in self._prefactors:
                yield prefactor, self._op
