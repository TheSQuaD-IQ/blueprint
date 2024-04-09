import math
from typing import Iterable, Tuple, Dict, List, Union, Callable, Iterator

from jax import Array
from jax import numpy as jnp
from jax.scipy.linalg import eigh

from ..base import QuantumSystem
from ..couplings import Coupling
from ..util.linalg import transform_op, tensor_product
from ..util.index import state_index, max_overlap_inds

Numeric = Union[float, complex]


class Device:
    """
    Device Base device class.
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

        self._qubits: List[QuantumSystem] = list(
            qubits
        )  # TODO: why is this a list and not a dict?
        self._qubit_inds: Dict[str, int] = {
            qubit.label: ind for ind, qubit in enumerate(qubits)
        }

        dim = math.prod(self.qubit_dims)
        self._dim: int = dim

        self._truncated: bool = False
        self._trunc_dim: int | None = None

        for ind, qubit in enumerate(self._qubits):
            qubit.embed(ind, self.qubit_dims)

        self._diagonalized: bool = False
        self._transform: Array | None = None

        self._eig_inds: Array | None = None

        self._couplings: Dict[str, Coupling] = {}

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

    def get_qubit_index(self, label: str) -> int:
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
    def qubits(self) -> Tuple[QuantumSystem, ...]:
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
    def qubit_dims(self) -> Tuple[int, ...]:
        """
        qubit_dims Returns the dimensions of the qubits in the device.

        Returns
        -------
        Tuple[int]
            The dimensions of the qubits in the device.
        """
        dims = tuple((qubit.truncated_dim or qubit.dim for qubit in self._qubits))
        return dims

    @property
    def qubit_labels(self) -> Tuple[str, ...]:
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
            self._trunc_dim = dim
        else:
            self._truncated = False
            self._trunc_dim = dim

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

    def add_capacative_coupling(
        self, qubits: Tuple[str, str], label: str, prefactor: Numeric
    ) -> None:
        """
        add_capacative_coupling Adds a capacitive coupling between a pair of qubit (specified by their labels) to the device. The coupling is specified by a prefactor and couples the labels via their charge operators. For more information about the coupling prefactor, see the documentation of the `Coupling` class.

        Parameters
        ----------
        qubits : Tuple[str, str]
            The labels of the pair of qubits to be coupled.
        label : str
            The label of the coupling term.
        prefactor : GenNumeric
            The prefactor of the coupling term. This can either be a constant factor (float or complext) or a callable that takes in parameters and returns a numeric value.
        """

        if not isinstance(qubits, tuple):
            raise ValueError(
                f"The qubit labels 'qubit_labels' expeted as a tuple, instead got type {type(qubits)}."
            )

        if len(qubits) != 2:
            raise ValueError(
                f"The qubit labels 'qubit_labels' expected as a tuple of length 2, instead got a tuple of length {len(qubits)}."
            )

        ops = []
        for ind, qubit in enumerate(qubits):
            if not isinstance(qubit, str):
                raise ValueError(
                    f"Each qubit label 'qubit_labels' expected as a string, instead got a type {type(qubit)} for the label at index {ind}."
                )
            if qubit not in self._qubit_inds:
                raise ValueError(f"Qubit {qubit} not found in the device.")

            qubit_ind = self._qubit_inds[qubit]

            try:
                op = self._qubits[qubit_ind].get_charge_op()
            except AttributeError as exc:
                raise AttributeError(
                    f"Qubit {qubits[ind]} does not have a charge operator."
                ) from exc

            ops.append(op)

        operator = jnp.matmul(ops[0], ops[1])

        coupling = Coupling(
            label=label,
            operator=operator,
            prefactor=prefactor,
            qubit_labels=qubits,
        )

        self._couplings[label] = coupling

    def get_eigenvalues(self) -> Array:
        """
        eig_vals Returns the eigenvalues of the qubit Hamiltonian.

        Returns
        -------
        Array
            The eigenvalues of the qubit Hamiltonian.
        """
        dim = self._trunc_dim or self._dim

        hamiltonian = self._get_hamiltonian()
        eig_vals = eigh(hamiltonian, eigvals_only=True)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals[:dim]

    def get_eigenstates(self, *, diagonalize: bool = True) -> Tuple[Array, Array]:
        """
        get_eigenstates Returns the eigenvalues and eigenvectors of the qubit Hamiltonian.

        Parameters
        ----------
        diagonalize : bool, optional
            Whether to return the diagonalized eigenstates, by default True

        Returns
        -------
        Tuple[Array, Array]
            The eigenvalues and eigenvectors of the qubit Hamiltonian.
        """
        dim = self._trunc_dim or self._dim

        if diagonalize and self._diagonalized:
            eig_vals = self.get_eigenvalues()
            eig_vecs = jnp.identity(dim)
            return eig_vals[:dim], eig_vecs

        hamiltonian = self._get_hamiltonian()
        eig_vals, eig_vecs = eigh(hamiltonian, eigvals_only=False)
        norm_vals = eig_vals - eig_vals[0]
        return norm_vals[:dim], eig_vecs[:, :dim]

    def _get_diagonal_hamiltonian(self) -> Array:
        dim = self._trunc_dim or self._dim
        eig_vals = self.get_eigenvalues()
        hamiltonian = jnp.diag(eig_vals[:dim])
        return hamiltonian

    def _get_bare_hamiltonian(self) -> Array:
        """
        _get_bare_hamiltonian Computes the bare Hamiltonian of the device.

        Returns
        -------
        Array
            The bare Hamiltonian of the device.
        """
        bare_hamiltonian = jnp.zeros((self._dim, self._dim))

        for qubit in self._qubits:
            bare_hamiltonian += qubit.get_hamiltonian()
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
            hamiltonian = self.process_op(diag_hamiltonian, diagonalize=False)
            return hamiltonian

        bare_hamiltonian = self._get_bare_hamiltonian()
        hamiltonian = self.process_op(bare_hamiltonian)
        return bare_hamiltonian

    def _get_int_hamiltonian(self) -> Array:
        """
        get_int_hamiltonian Returns the interaction Hamiltonian of the device.

        Returns
        -------
        Array
            The interaction Hamiltonian of the device.
        """
        int_hamiltonian = jnp.zeros((self._dim, self._dim))

        for coupling in self._couplings.values():
            int_hamiltonian += coupling.get_hamiltonian()
        return int_hamiltonian

    def get_int_hamiltonian(self) -> Array:
        """
        get_int_hamiltonian Returns the interaction Hamiltonian of the device.

        Returns
        -------
        Array
            The interaction Hamiltonian of the device.
        """
        native_hamiltonian = self._get_int_hamiltonian()
        hamiltonian = self.process_op(native_hamiltonian)
        return hamiltonian

    def _get_hamiltonian(self) -> Array:
        bare_hamiltonian = self._get_bare_hamiltonian()
        int_hamiltonian = self._get_int_hamiltonian()
        return bare_hamiltonian + int_hamiltonian

    def get_hamiltonian(self) -> Array:
        """
        get_himiltonian Returns the full Hamiltonian of the device.

        Returns
        -------
        Array
            The full Hamiltonian of the device.
        """
        native_hamiltonian = self._get_hamiltonian()
        hamiltonian = self.process_op(native_hamiltonian)
        return hamiltonian

    def _get_drive_hamiltonian(self, **params) -> Array:
        """
        _get_drive_hamiltonian Returns the sum of the Hamiltonian of each of the drives
        applied to the qubits and couplings of the device.

        Returns
        -------
        Array
            The total drive Hamiltonian of the device.
        """
        hamiltonian = jnp.zeros((self._dim, self._dim))

        for qubit in self.qubits:
            for drive in qubit._drives.values():
                drive_hamiltonian = drive.get_hamiltonian(**params)
                hamiltonian = jnp.add(hamiltonian, drive_hamiltonian)

        for coupling in self._couplings.values():
            for drive in coupling._drives.values():
                drive_hamiltonian = drive.get_hamiltonian(**params)
                hamiltonian = jnp.add(hamiltonian, drive_hamiltonian)

        return hamiltonian

    def get_drive_hamiltonian_terms(self, **params) -> Iterator[Tuple[Callable, Array]]:
        """
        get_drive_hamiltonian_terms Returns an iterator over the
        `(prefactor, processed_op)` for each drive applied to the qubits and couplings
        or the device.

        Returns
        -------
        Array
            The total drive Hamiltonian of the device.
        """
        for qubit in self.qubits:
            for drive in qubit._drives.values():
                for prefactor, op in drive.decompose(**params):
                    yield prefactor, self.process_op(op)

        for coupling in self._couplings.values():
            for drive in coupling._drives.values():
                for prefactor, op in drive.decompose(**params):
                    yield prefactor, self.process_op(op)

    def get_drive_hamiltonian(self, **params) -> Array:
        """
        get_drive_hamiltonian Returns the sum of the Hamiltonian of each of the drives
        applied to the device.

        Returns
        -------
        Array
            The total drive Hamiltonian of the device.
        """
        drive_hamiltonian = self._get_drive_hamiltonian(**params)
        return self.process_op(drive_hamiltonian)

    def diagonalize(self, truncated_dim: int | None = None) -> None:
        """
        diagonalize Diagonalizes the device Hamiltonian.

        Parameters
        ----------
        truncated_dim : int | None, optional
            The dimension by which to truncate the Hamiltonian, by default None

        Raises
        ------
        ValueError
            If the truncated dimension is not an integer.
        ValueError
            If the truncated dimension is less than 0 or greater than the current
            dimension.
        """
        if self.is_diagonalized:
            raise RuntimeError("The device has already been diagonalized.")

        if truncated_dim is not None:
            self.truncated_dim = truncated_dim

        _, eig_vecs = self.get_eigenstates()

        self._diagonalized = True
        self._transform = eig_vecs

    def process_op(
        self,
        op: Array,
        *,
        diagonalize: bool = True,
        truncate: bool = True,
    ) -> Array:
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
            if self._transform is None:
                raise ValueError("The transform matrix is not set.")
            op = transform_op(op, self._transform)

        if truncate and self._truncated:
            if self._trunc_dim is None:
                raise ValueError(
                    "The truncation dimension is not set, making it impossible to perform this."
                )
            op = op[: self._trunc_dim, : self._trunc_dim]

        return op

    def index_eigenstates(self) -> None:
        """
        index_eigenstates Indexes the eigenstates of the device by finding the maximum overlap with the bare states.
        """
        states_list = []
        for qubit in self._qubits:
            _, qubit_states = qubit.get_eigenstates()
            states_list.append(qubit_states)

        bare_states = tensor_product(states_list)
        _, dressed_states = self.get_eigenstates()

        self._eig_inds = max_overlap_inds(bare_states, dressed_states)

    def get_eigenstate(self, *state_indices: int) -> Tuple[Array, Array]:
        """
        get_eigenstate Returns the eigen energy and eigen state of a specific state of the device, as specified by an list of indices (integers). Each index corresponds to the the energy/number of excitations of each respective qubit.

        Parameters
        ----------
        state_index : Tuple[int]
            The index of the state, as determined by the energy/number of excitations of each qubit. The order of the qubits is the same as the order in which they were added to the device. Therefore, the length of the tuple must match the number of qubits in the device. Each index must be an integer and smaller than the dimension of the corresponding qubit. For example, for a device consisting of two qubits, the state [0, 1] corresponds to the ground state of the first qubit and the first excited state of the second qubit.

        Returns
        -------
        Tuple[Array, Array]
            The eigen energy and eigen state of the specified state.
        """
        state_ind = state_index(state_indices, self.qubit_dims)

        if self._eig_inds is None:
            raise ValueError(
                "Please index the eigenstates of the device by callling 'index_eigenstates' before accessing them."
            )

        eig_ind = self._eig_inds[state_ind]

        eig_vals, eig_vecs = self.get_eigenstates()
        energy = eig_vals[eig_ind]
        state = eig_vecs[:, eig_ind]
        return energy, state
