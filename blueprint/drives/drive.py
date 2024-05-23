from typing import Union, Callable, Iterable, List, Iterator, Tuple

from jax import Array
from jax import numpy as jnp

from ..util import Partial

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
        self._prefactors: List[Partial] = []

        if isinstance(prefactor, Callable):
            # partial_prefactor = Partial(prefactor)
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

                # partial_prefactor = Partial(op_prefactor)
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
    def prefactors(self) -> List[Partial]:
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

    @property
    def free_params(self) -> Tuple[str, ...]:
        """
        free_params Returns the free parameters of the possibly time-dependent drive term.

        Returns
        -------
        Tuple[str]
            The tuple of free parameters (str) of the possibly time-dependent drive term.
        """
        params = set()
        for prefactor in self._prefactors:
            params.update(prefactor.free_params)
        return tuple(params)

    @property
    def params(self) -> Tuple[str, ...]:
        """
        params Returns the parameters of the possibly time-dependent drive term. This includes both free and fixed parameters.

        Returns
        -------
        Tuple[str]
            The tuple of parameters (str) of the possibly time-dependent drive term.
        """
        params = set()
        for prefactor in self._prefactors:
            params.update(prefactor.params)
        return tuple(params)

    def set_params(self, **keywords) -> None:
        """
        set_params Sets the parameters of the time-dependent term.

        Parameters
        ----------
        **keywords
            The parameters of the time-dependent term.
        """
        for prefactor in self._prefactors:
            prefactor_params = set(prefactor.keyword_args)

            for param, value in keywords.items():
                if param in prefactor_params:
                    prefactor.set_keyword(param, value)

    def eval_prefactors(self, **params) -> Iterator[Numeric]:
        """
        eval_prefactor Evaluates the prefactor of the time-dependent term.

        Returns
        -------
        Numeric
            The evaluated prefactor.
        """
        for prefactor in self._prefactors:
            args = []
            for pos_arg in prefactor.pos_only_args:
                if pos_arg not in params:
                    raise ValueError(f"Missing required positional argument {pos_arg}.")
                arg_val = params[pos_arg]
                args.append(arg_val)

            keyword_args = prefactor.keyword_args
            keywords = {}
            for keyword_arg in keyword_args:
                if keyword_arg in params:
                    keywords[keyword_arg] = params[keyword_arg]

            prefactor_val = prefactor(*args, **keywords)
            yield prefactor_val

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

    def decompose(
        self, finalize: bool = True, **params
    ) -> Iterator[Tuple[Callable, Array]]:
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
        if finalize:
            prefactors = self._prefactors
            """
            for prefactor in self._prefactors:
                args = []
                for pos_arg in prefactor.pos_only_args:
                    if pos_arg not in params:
                        raise ValueError(
                            f"Missing required positional argument {pos_arg}."
                        )
                    arg_val = params[pos_arg]
                    args.append(arg_val)

                keyword_args = prefactor.keyword_args
                keywords = {}
                for keyword_arg in keyword_args:
                    if keyword_arg in params:
                        keywords[keyword_arg] = params[keyword_arg]

                # finalized_prefactor = prefactor.finalize(*args, **keywords)
                prefactors.append(prefactor)
                """
        else:
            prefactors = self._prefactors

        if isinstance(self._op, list):
            yield from zip(prefactors, self._op)
        else:
            for prefactor in prefactors:
                yield prefactor, self._op
