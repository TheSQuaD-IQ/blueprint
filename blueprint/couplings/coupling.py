from typing import Union, Callable, Tuple, Dict, Iterator
from functools import wraps

from jax import Array
from jax import numpy as jnp

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

        self._dim: int = op_dim

        self._qubit_labels: Tuple[str, str] | None = qubit_labels

        self._truncated: bool = False
        self._trunc_dim: int | None = None

        self._diagonalized: bool = False
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

    @label.setter
    def label(self, label: str) -> None:
        """
        label Sets the label of the time-dependent term.

        Parameters
        ----------
        label : str
            The label of the time-dependent term.
        """
        if not isinstance(label, str):
            raise ValueError(
                f"The label must be a string, instead got type {type(label)}."
            )
        self._label = label

    @property
    def num_drives(self) -> int:
        """
        num_drives Returns the number of drives applied to the transmon.

        Returns
        -------
        int
            The number of drives applied to the transmon.
        """
        return len(self._drives)

    @property
    def drives(self) -> Dict[str, Drive]:
        """
        drives Returns the dictionary of drive labels and corresponding drives that have been applied to the quantum system.

        Returns
        -------
        Dict[str, Drive]
            The dictionary of drive labels and corresponding to each applied drive.
        """
        return self._drives

    @property
    def is_driven(self) -> bool:
        """
        is_driven Returns whether the transmon is driven.

        Returns
        -------
        bool
            True if the transmon is driven, False otherwise.
        """
        return any(self._drives)

    @property
    def is_truncated(self) -> bool:
        """
        is_truncated Returns whether the quantum system has been truncated.

        Returns
        -------
        bool
            Whether the quantum system has been truncated.
        """
        return self._truncated

    @property
    def is_diagonalized(self) -> bool:
        """
        is_diagonalized Returns whether the quantum system has been diagonalized.

        Returns
        -------
        bool
            Whether the quantum system has been diagonalized.
        """
        return self._diagonalized

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
    def truncated_dim(self) -> int | None:
        """
        truncated_dim Returns the dimension of the truncated Hilbert space.

        Returns
        -------
        int
            The dimension of the truncated Hilbert space. If None, the Hilbert space is not truncated.
        """
        return self._trunc_dim

    @truncated_dim.setter
    def truncated_dim(self, dim: int | None) -> None:
        """
        truncated_dim Sets the dimension of the truncated Hilbert space.

        Parameters
        ----------
        dim : int | None
            The dimension of the truncated Hilbert space. If None, the truncation is removed.
        """
        if dim is not None:
            if not isinstance(dim, int):
                raise ValueError(
                    "The dimension of the Hilbert space must be an integer."
                )
            if dim <= 0 or dim > self._dim:
                raise ValueError(
                    f"The Hilbert space dimension ('truncated_dim') must be greater than 0 and less than or equal to the current dimension ({self._dim})."
                )
            if dim == self._dim:
                self._truncated = False
            else:
                self._truncated = True
        else:
            self._truncated = False

        self._trunc_dim = dim

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

    def get_hamiltonian(self) -> Array:
        """
        get_hamiltonian Returns the Hamiltonian of the coupling term.

        Returns
        -------
        Array
            The coupling term Hamiltonian.
        """
        hamiltonian = self._prefactor * self._operator
        return self.process_op(hamiltonian)

    def _get_drive_hamiltonian(self, time: float) -> Array:
        hamiltonian = jnp.zeros((self._dim, self._dim))

        for drive in self._drives.values():
            drive_hamiltonian = drive.get_hamiltonian(time)
            hamiltonian = hamiltonian + drive_hamiltonian
        return hamiltonian

    def get_drive_hamiltonian_terms(self) -> Iterator[Tuple[Callable, Array]]:
        """
        get_drive_hamiltonian_terms

        Returns
        -------
        Array
            The total drive Hamiltonian.
        """
        for drive in self._drives.values():
            for prefactor, op in drive.decompose():
                yield prefactor, self.process_op(op)

    def get_drive_hamiltonian(self, time: float) -> Array:
        """
        get_drive_hamiltonian Returns the Hamiltonian of the drive terms of the coupler.

        Returns
        -------
        Array
            The drive Hamiltonian.
        """
        drive_hamiltonian = self._get_drive_hamiltonian(time)
        return self.process_op(drive_hamiltonian)

    def set_transform(self, transform: Array) -> None:
        """
        set_transform Sets the transformation matrix applied to the coupling.

        Parameters
        ----------
        transform : Array
            _description_
        """
        self._diagonalized = True
        self._transform = transform

    def process_op(
        self,
        op: Array,
        *,
        diagonalize: bool = True,
        truncate: bool = True,
    ) -> Array:
        """
        process_op Processes an operator expressed in the default coupler basis to another basis.
        This can include both transformation to the diagonal energy basis of the device
        and a truncation of the operator if the dimension of the hilbert space has been truncated.

        Parameters
        ----------
        op : Array
            The operator to be processed.
        diagonalize : bool, optional
            Whether to diagonalize the operator, by default True
        truncate : bool, optional
            Whether to truncate the operator, by default True

        Returns
        -------
        Array
            The processed operator.

        Raises
        ------
        ValueError
            If the transformation matrix is not set, but the operator has to be transformed.
        ValueError
            If the truncation dimension is not set, but the operator has to be truncated.
        """
        if diagonalize and self._diagonalized:
            if self._transform is None:
                raise ValueError(
                    "The transform matrix is not set, making it impossible to perform this."
                )
            op = transform_op(op, self._transform)

        if truncate and self._truncated:
            if self._trunc_dim is None:
                raise ValueError(
                    "The truncation dimension is not set, making it impossible to perform this."
                )
            op = op[: self._trunc_dim, : self._trunc_dim]
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
