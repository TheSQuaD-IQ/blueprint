from typing import Union, Callable, Tuple, Dict, Iterator
from functools import wraps
from jax import Array
from jax import numpy as jnp

from ..base.terms import TimeDependentTerm
from ..drives import Drive
from ..util.linalg import transform_op

Numeric = Union[float, complex]


class Coupling:
    """
    Coupling A class for representing a coupling in the Hamiltonian.
    """

    def __init__(
        self,
        label: str,
        operator: Array,
        prefactor: Numeric,
        qubit_labels: Tuple[str, str] | None = None,
    ) -> None:
        if not isinstance(label, str):
            raise ValueError(
                f"The label must be a string, instead got type {type(label)}."
            )
        self._label = label

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

        self._prefactor: float | complex = prefactor
        self._operator: Array = operator

        self._native_dim: int = op_dim
        self._dim: int = op_dim

        self._qubit_labels: Tuple[str, str] | None = qubit_labels

        self._is_diagonalized: bool = False
        self._transform: Array | None = None

        self._drives: Dict[str, Drive] = {}

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
    def native_dim(self) -> int:
        """
        native_dim Returns the dimension of the Hilbert space of the coupling in the original basis (excluding any truncation when it is diagonalized).

        Returns
        -------
        int
            The dimension of the Hilbert space of the quantum system in the native basis.
        """
        return self._native_dim

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
    def qubit_labels(self) -> Tuple[str, str] | None:
        """
        qubit_labels Returns the labels of the qubits in the coupling term.

        Returns
        -------
        Tuple[str, str] | None
            The labels of the qubits in the coupling term.
        """
        return self._qubit_labels

    @property
    def is_diagonalized(self) -> bool:
        """
        is_diagonalized Returns whether the coupling term is diagonalized.

        Returns
        -------
        bool
            True if the coupling term is diagonalized, False otherwise.
        """
        return self._is_diagonalized

    def get_hamiltonian(self) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian of the coupling term.

        Returns
        -------
        Array
            The coupling term Hamiltonian.
        """
        native_hamiltonian = self._prefactor * self._operator
        hamiltonian = self.process_op(native_hamiltonian)
        return hamiltonian

    def _get_drive_hamiltonian(self, **params) -> Array:
        hamiltonian = jnp.zeros((self._native_dim, self._native_dim))

        for drive in self._drives.values():
            drive_hamiltonian = drive.get_hamiltonian(**params)

            hamiltonian = jnp.add(hamiltonian, drive_hamiltonian)
        return hamiltonian

    def get_drive_hamiltonian(
        self, decompose: bool = False, **params
    ) -> Array | Iterator[Tuple[TimeDependentTerm, Array]]:
        """
        get_drive_hamiltonian Returns the Hamiltonian of the drive terms.

        Parameters
        ----------
        decompose : bool, optional
            Whether to decompose this term into a series of time-dependent prefactors and operators, by default False

        Returns
        -------
        Array | Iterator[Tuple[TimeDependentTerm, Array]]
            _description_

        Yields
        ------
        Iterator[Array | Iterator[Tuple[TimeDependentTerm, Array]]]

        """
        if decompose:
            for drive in self._drives.values():
                for prefactor, op in drive.decompose():
                    yield prefactor, self.process_op(op)

        else:
            drive_hamiltonian = self._get_drive_hamiltonian(**params)
            return self.process_op(drive_hamiltonian)

    def set_transform(self, transform: Array) -> None:
        """
        set_transform Sets the transformation matrix applied to the coupling.

        Parameters
        ----------
        transform : Array
            _description_
        """
        self._dim = transform.shape[0]
        self._is_diagonalized = True
        self._transform = transform

    def process_op(self, op: Array, *, diagonalize: bool = True) -> Array:
        """
        process_op Processes an operator in the native system basis to the transformed basis.
        This can include both transformation to the diagonalized basis and/or
        an embedding of the operator in a higher-dimensional Hilbert space.

        Parameters
        ----------
        op : Array
            The operator expressed in the native basis used by the quantum system.
        diagonalize : bool, optional
            Whether to diagonalize the operator, by default True
        embed : bool, optional
            Whether to embed the operator in a higer-dimensional Hilbert space, by default True

        Returns
        -------
        Array
            _description_
        """
        if self.is_diagonalized and diagonalize:
            if self._transform is None:
                raise ValueError("The transform matrix is not set.")
            op = transform_op(op, self._transform)
        return op

    def add_drive(self, label: str, coupling_pulse: Callable) -> None:
        """
        add_drive Adds a drive to the coupling term.

        Parameters
        ----------
        label : str
            The label of the drive.
        coupling_pulse : Callable
            The time-dependent coupling strength pulse resulting from the application of the drive. This must be a callable object that the coupling strength at a given time.

        Raises
        ------
        ValueError
            If a drive with the same label has already been added to the coupling term.
        """
        if label in self._drives:
            raise ValueError(
                f"A drive with the label '{label}' has alreadted been applied to the transmon."
            )

        @wraps(coupling_pulse)
        def prefactor(*args, **kwargs) -> float:
            coup_prefactor = coupling_pulse(*args, **kwargs)
            return coup_prefactor - self._prefactor

        drive = Drive(label, prefactor, self._operator)
        self._drives[label] = drive
