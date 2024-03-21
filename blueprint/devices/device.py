import math
from typing import Iterable, Tuple, Dict, List, Union, Callable
from functools import reduce

from jax import Array
from jax import numpy as jnp
from jax.scipy.linalg import eigh

from ..base import QuantumSystem
from ..couplings import Coupling
from ..util.linalg import transform_op

Numeric = Union[float, complex]
GenNumeric = Union[Numeric, Callable[..., Numeric]]


class Device:
    """
    Device Base device class.
    #TODO: Both the Qubit and Device classes shouble be subclasses of some base class QuantumSystem, which implements eigenvalue/diagonalization functionality. However, I am not sure if it makes sense to have the Device support embed functionality. Keeping it as a separate class for now.
    """

    def __init__(self, qubits: Iterable[QuantumSystem]) -> None:
        labels = set()

        for qubit in qubits:
            if not isinstance(qubit, QuantumSystem):
                raise ValueError(
                    "Each qubit in the device must be instances of the QuantumSystem class, "
                    f"instead got type {type(qubit)}."
                )

            label = qubit.label
            if label in labels:
                raise ValueError(
                    f"Qubit {label} already added to the device. "
                    "Please ensure that all qubit labels are unique."
                )

            labels.add(label)

            if qubit.is_embedded:
                raise ValueError(
                    f"Qubit {label} is already embedded into a larger Hilbert space."
                    "Please ensure that qubits are not already part of a device."
                )

        self._qubits: List[QuantumSystem] = list(qubits)
        self._qubit_inds: Dict[str, int] = {
            qubit.label: ind for ind, qubit in enumerate(qubits)
        }

        self._dim: int = math.prod(self.qubit_dims)

        for ind, qubit in enumerate(self._qubits):
            qubit.embed(ind, self.qubit_dims)

        self._couplings: List[Coupling] = []

        self._diagonalized: bool = False
        self._transform: Array = None

    def __getitem__(self, label: str) -> QuantumSystem:
        if not isinstance(label, str):
            raise ValueError(
                f"The qubit label must be a string, instead got type {type(label)}."
            )

        try:
            ind = self._qubit_inds[label]
        except KeyError as exc:
            raise ValueError(f"Qubit {label} not found in the device.") from exc

        return self._qubits[ind]

    def __contains__(self, label: str) -> bool:
        if not isinstance(label, str):
            raise ValueError(
                f"The qubit label must be a string, instead got type {type(label)}."
            )

        return label in self._qubit_inds

    def __len__(self) -> int:
        return self.num_qubits

    def __iter__(self) -> Iterable[QuantumSystem]:
        yield from self._qubits

    def get_index(self, label: str) -> int:
        """
        get_index Returns the index of the qubit in the device.

        Parameters
        ----------
        label : str
            The label of the qubit.

        Returns
        -------
        int
            The index of the qubit in the device.

        Raises
        ------
        ValueError
            If the qubit label is not a string.
        ValueError
            If the qubit label is not found in the device.
        """
        if not isinstance(label, str):
            raise ValueError(
                f"The qubit label must be a string, instead got type {type(label)}."
            )

        try:
            ind = self._qubit_inds[label]
        except KeyError as exc:
            raise ValueError(f"Qubit {label} not found in the device.") from exc

        return ind

    @property
    def is_transformed(self) -> bool:
        """
        transformed Returns whether the device is transformed into a new basis.

        Returns
        -------
        bool
            Whether the device is transformed into a new basis.
        """
        return self._transform is not None

    @property
    def qubits(self) -> Tuple[QuantumSystem]:
        """
        qubits Returns the qubits in the device.

        Returns
        -------
        Tuple[QuantumSystem]
            The tuple of qubits in the device.
        """
        return tuple(self._qubits)

    @property
    def num_qubits(self) -> int:
        """
        num_qubits Returns the number of qubits in the device.

        Returns
        -------
        int
            The number of qubits in the device.
        """
        return len(self._qubits)

    @property
    def qubit_dims(self) -> Tuple[int]:
        """
        qubit_dims Returns the dimensions of the qubits in the device.

        Returns
        -------
        Tuple[int]
            The dimensions of the qubits in the device.
        """
        dims = tuple((qubit.dim for qubit in self._qubits))
        return dims

    @property
    def qubit_labels(self) -> Tuple[str]:
        """
        qubit_labels Returns the labels of the qubits in the device.

        Returns
        -------
        Tuple[str]
            The labels of the qubits in the device.
        """
        labels = tuple(self._qubit_inds)
        return labels

    @property
    def dim(self) -> int:
        """
        hilbert_dim Returns the dimension of the Hilbert space of the device.

        Returns
        -------
        int
            The dimension of the Hilbert space of the device.
        """
        return self._dim

    @property
    def is_diagonalized(self) -> bool:
        """
        is_diagonalized Returns whether the device Hamiltonian has been diagonalized.

        Returns
        -------
        bool
            Whether the device Hamiltonian has been diagonalized.
        """
        return self._diagonalized

    def add_capacative_coupling(
        self, qubit_labels: Tuple[str, str], coupler_label: str, prefactor: GenNumeric
    ) -> None:
        """
        add_capacative_coupling Adds a capacitive coupling between a pair of qubit (specified by their labels) to the device. The coupling is specified by a prefactor and couples the labels via their charge operators. For more information about the coupling prefactor, see the documentation of the `Coupling` class.

        Parameters
        ----------
        qubit_labels : Tuple[str, str]
            The labels of the pair of qubits to be coupled.
        coupler_label : str
            The label of the coupling term.
        prefactor : GenNumeric
            The prefactor of the coupling term. This can either be a constant factor (float or complext) or a callable that takes in parameters and returns a numeric value.
        """

        if not isinstance(qubit_labels, tuple):
            raise ValueError(
                f"The qubit labels 'qubit_labels' expeted as a tuple, instead got type {type(qubit_labels)}."
            )
        if len(qubit_labels) != 2:
            raise ValueError(
                f"The qubit labels 'qubit_labels' expected as a tuple of length 2, instead got a tuple of length {len(qubit_labels)}."
            )
        for ind, label in enumerate(qubit_labels):
            if not isinstance(label, str):
                raise ValueError(
                    f"Each qubit label 'qubit_labels' expected as a string, instead got a type {type(label)} for the label at index {ind}."
                )
            if label not in self._qubit_inds:
                raise ValueError(f"Qubit {label} not found in the device.")

        inds = tuple((self._qubit_inds[label] for label in qubit_labels))
        ops = []

        for ind in inds:
            try:
                op = self._qubits[ind].get_charge_op()
            except AttributeError as exc:
                raise AttributeError(
                    f"Qubit {qubit_labels[ind]} does not have a charge operator."
                ) from exc

            ops.append(op)

        operator = reduce(jnp.matmul, ops)

        dims = tuple((self._qubits[ind].dim for ind in inds))

        coupling = Coupling(
            label=coupler_label,
            operator=operator,
            prefactor=prefactor,
            dims=dims,
            qubit_labels=qubit_labels,
        )

        self._couplings.append(coupling)

    def eigenvalues(self, **kwargs) -> Array:
        """
        eig_vals Returns the eigenvalues of the qubit Hamiltonian.

        Returns
        -------
        Array
            The eigenvalues of the qubit Hamiltonian.
        """
        hamiltonian = self._get_bare_hamiltonian()
        eig_vals = eigh(hamiltonian, eigvals_only=True, **kwargs)
        return eig_vals

    def eigenstates(self, **kwargs) -> Tuple[Array, Array]:
        """
        eig_sys Returns the eigenvalues and eigenvectors of the qubit Hamiltonian.

        Returns
        -------
        Tuple[Array, Array]
            The eigenvalues and eigenvectors of the qubit Hamiltonian.
        """
        hamiltonian = self._get_bare_hamiltonian()
        eig_vals, eig_vecs = eigh(hamiltonian, eigvals_only=False, **kwargs)
        return eig_vals, eig_vecs
    
    def _get_bare_hamiltonian(self) -> Array:
        """
        _get_bare_hamiltonian Computes the bare Hamiltonian of the device.

        Returns
        -------
        Array
            The bare Hamiltonian of the device.
        """
        qubit_hamiltonians = (qubit.get_hamiltonian() for qubit in self._qubits)
        bare_hamiltonian = reduce(jnp.add, qubit_hamiltonians)
        return bare_hamiltonian

    def get_bare_hamiltonian(self) -> Array:
        """
        get_bare_hamiltonian Returns the bare Hamiltonian of the device.

        Returns
        -------
        Array
            The bare Hamiltonian of the device.
        """
        if self.is_diagonalized:
            diag_hamiltonian = self._get_diagonal_hamiltonian()
            return diag_hamiltonian

        bare_hamiltonian = self._get_bare_hamiltonian()
        return bare_hamiltonian

    def get_int_hamiltonian(self, **params) -> Array:
        """
        get_int_hamiltonian Returns the interaction Hamiltonian of the device.

        Returns
        -------
        Array
            The interaction Hamiltonian of the device.
        """
        int_hamiltonian = jnp.zeros((self._dim, self._dim), dtype=jnp.complex64)
        for coupling in self._couplings:
            coup_hamiltonian = coupling.get_hamiltonian(**params)
            int_hamiltonian += coup_hamiltonian
        return int_hamiltonian

    def get_hamiltonian(self) -> Array:
        """
        get_himiltonian Returns the full Hamiltonian of the device.

        Returns
        -------
        Array
            The full Hamiltonian of the device.
        """
        bare_hamil = self.get_bare_hamiltonian()

        return bare_hamil

    def diagonalize(
        self, truncated_dim: int | None = None, *, sub_ground_energy: bool = True
    ) -> None:
        """
        diagonalize Diagonalizes the device Hamiltonian.

        Parameters
        ----------
        truncated_dim : int | None, optional
            The dimension by which to truncate the Hamiltonian, by default None
        sub_ground_energy : bool, optional
            Whether to subtract the ground state energy from the eigenvalues, by default True

        Raises
        ------
        ValueError
            If the truncated dimension is not an integer.
        ValueError
            If the truncated dimension is less than 0 or greater than the current dimension.
        """
        if self.is_diagonalized:
            raise RuntimeError("The device has already been diagonalized.")

        if truncated_dim is not None:
            if not isinstance(truncated_dim, int):
                raise ValueError(
                    "The dimension of the Hilbert space must be an integer."
                )
        if truncated_dim <= 0 or truncated_dim > self._dim:
            raise ValueError(
                f"The Hilbert space dimension ('truncated_dim') must be greater than 0 and less than or equal to the current dimension ({self._dim})."
            )

        eig_vals, eig_vecs = self.eigenstates()
        trunc_vals = eig_vals[:truncated_dim]
        if sub_ground_energy:
            trunc_vals = trunc_vals - trunc_vals[0]

        trunc_vecs = eig_vecs[:, :truncated_dim]

        self._diagonalized = True
        self._transform = trunc_vecs
        self._dim = truncated_dim

    def process_op(self, op: Array, *, diagonalize: bool = True) -> Array:
        """
        process_op Processes the native operator into the transformed basis.

        Parameters
        ----------
        native_op : Array
            The native operator.

        Returns
        -------
        Array
            The operator in the transformed basis.
        """
        if self.is_diagonalized and diagonalize:
            # Handle the case where the qubit implements the operators in an already diagonalized basis.
            if self._transform is not None:
                op = transform_op(op, self._transform)
        return op
