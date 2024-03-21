from typing import Union, Callable, Tuple
import math

from jax import Array

from ..base.terms import TimeDependentTerm

Numeric = Union[float, complex]
GenNumeric = Union[Numeric, Callable[..., Numeric]]


class Coupling:
    """
    Coupling A class for representing a coupling in the Hamiltonian.
    """
    def __init__(
        self,
        label: str,
        operator: Array,
        prefactor: GenNumeric,
        dims: Tuple[int, int] | None = None,
        qubit_labels: Tuple[str, str] | None = None,
    ) -> None:
        if not isinstance(label, str):
            raise ValueError(
                f"The label must be a string, instead got type {type(label)}."
            )
        self._label = label

        # TODO: potentially it might be better to provide the operator as a method that returns it, instead of saving it.
        if not isinstance(operator, Array):
            raise ValueError(
                f"The operator must be a jax.Array, instead got type {type(operator)}."
            )

        num_dims = len(operator.shape)
        if len(operator.shape) != 2:
            raise ValueError(
                f"The operator must be a 2D array, instead got a {num_dims}-dimensional array."
            )

        op_dim, _op_dim = operator.shape
        if op_dim != _op_dim:
            raise ValueError(
                f"The operator must be a square matrix, instead got a ({op_dim}x{_op_dim}) matrix."
            )

        if dims is not None:
            if not isinstance(dims, tuple):
                raise ValueError(
                    f"The dims must be a tuple, instead got type {type(dims)}."
                )
            if len(dims) != 2:
                raise ValueError(
                    f"The dims must be a tuple of length 2, instead got a tuple of length {len(dims)}."
                )
            for dim in dims:
                if not isinstance(dim, int):
                    raise ValueError(
                        f"The dims must be a tuple of integers, instead got a tuple of types {[type(dim) for dim in dims]}."
                    )

            if qubit_labels is not None:
                if not isinstance(qubit_labels, tuple):
                    raise ValueError(
                        f"The qubit_labels must be a tuple, instead got type {type(qubit_labels)}."
                    )
                if len(qubit_labels) != 2:
                    raise ValueError(
                        f"The qubit_labels must be a tuple of length 2, instead got a tuple of length {len(qubit_labels)}."
                    )
                for label in qubit_labels:
                    if not isinstance(label, str):
                        raise ValueError(
                            f"The qubit_labels must be a tuple of strings, instead got a tuple of types {[type(label) for label in qubit_labels]}."
                        )

        self._operator = operator
        self._dim = op_dim
        self._dims = dims
        self._qubit_labels = qubit_labels

        self._prefactor = TimeDependentTerm(prefactor)

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
    def qubit_dims(self) -> Tuple[int, int] | None:
        """
        qubit_dims Returns the dimensions of the qubits in the coupling term.

        Returns
        -------
        Tuple[int, int] | None
            The dimensions of the qubits in the coupling term.
        """
        return self._dims

    @property
    def qubit_labels(self) -> Tuple[str, str] | None:
        """
        qubit_labels Returns the labels of the qubits in the coupling term.

        Returns
        -------
        Tuple[str, str] | None
            The labels of the qubits in the coupling term.
        """
        return self._qubit_labels

    def set_params(self, **params) -> None:
        """
        set_params Sets the parameters of the time-dependent term.

        Parameters
        ----------
        **params
            The parameters of the time-dependent term.
        """
        self._prefactor.set_params(**params)

    def eval_prefactor(self, **params) -> Numeric:
        """
        eval_prefactor Evaluates the prefactor of the time-dependent term.

        Returns
        -------
        Numeric
            The evaluated prefactor.
        """
        return self._prefactor(**params)

    def _get_hamiltonian(self, *, exclude_prefactor: bool = False, **params) -> Array:
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
        if exclude_prefactor:
            return self._operator

        prefactor = self._prefactor(**params)
        if not isinstance(prefactor, (float, complex)):
            raise ValueError(
                f"The evaluated prefactor must be a number (float or complex), instead got type {type(prefactor)}."
            )

        return prefactor * self._operator

    def get_hamiltonian(self, *, exclude_prefactor: bool = False, **params) -> Array:
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
        hamiltonian = self._get_hamiltonian(
            exclude_prefactor=exclude_prefactor, **params
        )
        # TODO: consider if we want this term to also have operator processing methods, like transform or expand - in that case this would a subclass of QuantumSystem as well. If not, maybe merge this method with _get_hamilotinan.

        return hamiltonian
