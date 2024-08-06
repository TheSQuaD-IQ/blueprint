"""This module contains the Circuit class, which is used to represent a quantum circuit."""

from __future__ import annotations
import warnings
from itertools import chain
from typing import Callable, Iterable, Iterator, List, Self, Tuple

from ..util.runtime_checks import all_equal

VALID_TYPES = {"charge", "flux", "detuning"}


class Operation:
    """
    _summary_: A class to represent a quantum operation, implemented by a group of pulses applied on the qubits.
    """

    def __init__(
        self,
        name: str,
        qubits: str | Iterable[str],
        drive_pulses: Callable | Iterable[Callable],
        drive_types: str | Iterable[str],
        duration: float,
        time: float = 0.0,
    ):
        """
        __init__ Initializes the Operation object.

        Parameters
        ----------
        name : str
            The name of the operation.
        qubits : str | Iterable[str]
            The qubits on which the operation is applied.
        drive_pulses : Callable | Iterable[Callable]
            The pulses that implement the operation.
        drive_types : str | Iterable[str]
            The type of drive for each qubit. Must be one of "charge", "flux", or "detuning".
        duration : float
            The duration of the operation.
        time : float, optional
            The starting time of the operation, by default 0.0

        Raises
        ------
        ValueError
            If name is not a string.
        ValueError
            If qubits is not a string or an iterable of strings.
        ValueError
            If drive_pulses is not a callable or an iterable of callables.
        ValueError
            If drive_types is not a string or an iterable of strings.
        ValueError
            If duration is not a float.
        ValueError
            If duration is less than or equal to 0.
        ValueError
            If time is not a float.
        ValueError
            If time is less than 0.
        ValueError
            If the number of qubits does not match the number of pulses.
        ValueError
            If the number of qubits does not match the number of drive_types.
        ValueError
            If a drive type is not one of "charge", "flux", or "detuning".
        """

        if not isinstance(name, str):
            raise ValueError("name must be a str")
        self._name: str = name

        if isinstance(qubits, str):
            self._qubits = (qubits,)
        else:
            try:
                self._qubits = tuple(qubits)
            except TypeError as err:
                raise ValueError("qubits must be an iterable of str") from err

            for qubit in self._qubits:
                if not isinstance(qubit, str):
                    raise ValueError("Each element in qubits must be str")

        num_qubits = len(self._qubits)

        if isinstance(drive_pulses, Callable):
            self._pulses = (drive_pulses,)
        else:
            try:
                self._pulses = tuple(drive_pulses)
            except TypeError as err:
                raise ValueError(
                    "operations must be an iterable of drive_pulses"
                ) from err

            for pulse in self._pulses:
                if not isinstance(pulse, Callable):
                    raise ValueError(
                        "Each element of drive_pulses must be a Pulse object"
                    )

        num_pulses = len(self._pulses)
        if num_qubits != num_pulses:
            raise ValueError(
                f"Number of qubits ({num_qubits}) must match number of pulses ({num_pulses})"
            )

        if isinstance(drive_types, str):
            self._types = (drive_types,)
        else:
            try:
                self._types = tuple(drive_types)
            except TypeError as err:
                raise ValueError("drive_types must be an iterable of strings") from err

            for drive_type in self._types:
                if not isinstance(drive_type, str):
                    raise ValueError("Each element in drive_types must be a str")

                if drive_type not in VALID_TYPES:
                    valid_types_str = ", ".join(VALID_TYPES)
                    raise ValueError(
                        f"{drive_type} is not a valid drive type. Valid types are: {valid_types_str}"
                    )

        num_types = len(self._types)
        if num_qubits != num_types:
            raise ValueError(
                f"Number of qubits ({num_qubits}) must match number of drive_types ({num_types})"
            )

        if not isinstance(time, float):
            raise ValueError("time must be a float")

        if time < 0:
            raise ValueError("time must be greater than or equal to 0")

        self._time = time

        if not isinstance(duration, float):
            raise ValueError("duration must be a float")

        if duration <= 0:
            raise ValueError("duration must be greater than 0")

        self._duration = duration

    def __eq__(self, other: Self) -> bool:
        """
        __eq__ Compares the Operation object with another object.

        Parameters
        ----------
        other : Operation
            The object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if isinstance(other, Operation):
            attr = ("name", "qubits", "pulses", "pulse_types", "time", "duration")
            for attr_name in attr:
                if getattr(self, attr_name) != getattr(other, attr_name):
                    return False
        return False

    def __ne__(self, other: Self) -> bool:
        return not self.__eq__(other)

    def __copy__(self) -> Self:
        """
        __copy__ Creates a copy of the Operation object.

        Returns
        -------
        Self
            A shallow copy of the Operation object.
        """
        operation_copy = self.__class__(
            name=self._name,
            qubits=self._qubits,
            drive_pulses=self._pulses,
            drive_types=self._types,
            duration=self._duration,
            time=self._time,
        )
        return operation_copy

    def __str__(self) -> str:
        """
        __str__ Returns a string representation of the Operation object.

        Returns
        -------
        str
            The string representation of the Operation object.
        """
        qubits_str = ", ".join(self.qubits)
        repr_str = f"{self.name}({qubits_str})"
        return repr_str

    def __repr__(self) -> str:
        """
        __repr__ Returns a string representation of the Operation object.

        Returns
        -------
        str
            The string representation of the Operation object.
        """
        attributes = ("name", "qubits", "pulses", "pulse_types", "time", "duration")
        attribute_str_iter = (f"{attr}={getattr(self, attr)!r}" for attr in attributes)
        attributes_strs = ", ".join(attribute_str_iter)

        cls_name = self.__class__.__name__
        repr_str = f"{cls_name}({attributes_strs})"
        return repr_str

    def __add__(self, other: Self | Layer) -> Layer:
        """
        __add__ Adds an operation or a layer to the current operation.

        Parameters
        ----------
        other : Operation | Layer
            The operation to concatenate with.

        Returns
        -------
        Operation
            The concatenated operation.
        """
        if isinstance(other, Operation):
            try:
                layer = Layer(self, other)
            except ValueError as err:
                raise ValueError("Cannot add the two operations together") from err
            return layer
        elif isinstance(other, Layer):
            ops = (self, *other.operations)
            try:
                layer = Layer(*ops)
            except ValueError as err:
                raise ValueError("Cannot add the operation to the layer") from err
            return layer

        raise ValueError("operation must be an Operation object")

    def __radd__(self, other: Self | Layer) -> Layer:
        """
        __radd__ Adds an operation or a layer to the current operation.

        Parameters
        ----------
        other : Operation | Layer
            The operation to concatenate with.

        Returns
        -------
        Operation
            The concatenated operation.
        """
        if isinstance(other, Operation):
            try:
                layer = Layer(other, self)
            except ValueError as err:
                raise ValueError("Cannot add the two operations together") from err
            return layer
        elif isinstance(other, Layer):
            ops = (*other.operations, self)
            try:
                layer = Layer(*ops)
            except ValueError as err:
                raise ValueError("Cannot add the operation to the layer") from err
            return layer

        raise ValueError("operation must be an Operation object")

    @property
    def name(self) -> str:
        """
        name Returns the name of the operation.

        Returns
        -------
        str
            The name of the operation.
        """
        return self._name

    @property
    def qubits(self) -> Tuple[str, ...]:
        """
        qubits Returns the qubits on which the operation is applied.

        Returns
        -------
        Tuple[str, ...]
            The qubits on which the operation is applied.
        """
        return self._qubits

    @property
    def num_qubits(self) -> int:
        """
        num_qubits Returns the number of qubits on which the operation is applied.

        Returns
        -------
        int
            The number of qubits on which the operation is applied.
        """
        return len(self._qubits)

    @property
    def time(self) -> float:
        """
        time Returns the starting time of the operation.

        Returns
        -------
        float
            The starting time of the operation.
        """
        return self._time

    @time.setter
    def time(self, new_time: float) -> None:
        """
        time Sets the starting time of the operation.

        Parameters
        ----------
        new_time : float
            The new starting time of the operation.

        Raises
        ------
        ValueError
            If new_time is not a float
        """
        if not isinstance(new_time, float):
            raise ValueError("new_time must be a float")
        self._time = new_time

    @property
    def duration(self) -> float:
        """
        duration Returns the duration of the operation.

        Returns
        -------
        float
            The duration of the operation.
        """
        return self._duration

    @duration.setter
    def duration(self, new_duration: float) -> None:
        """
        duration Sets the duration of the operation.

        Parameters
        ----------
        new_duration : float
            The new duration of the operation.
            Note that this does not modify the pulses.
            In other words, it assumes that the pulses return 0s outside of the
            times during which they are applied.

        Raises
        ------
        ValueError
            If new_duration is not a float
        """
        if not isinstance(new_duration, float):
            raise ValueError("new_duration must be a float")
        self._duration = new_duration

    @property
    def pulses(self) -> Tuple[Callable, ...]:
        """
        pulses Returns the pulses that implement the operation.

        Returns
        -------
        Tuple[Callable, ...]
            The pulses that implement the operation.
        """
        return self._pulses

    @property
    def pulse_types(self) -> Tuple[str, ...]:
        """
        pulse_types Returns the types of drive for each qubit.

        Returns
        -------
        Tuple[str, ...]
            The types of drive for each qubit.
        """
        return self._types

    @property
    def drive_label(self) -> str:
        """
        drive_label Returns the label for the drive.

        Returns
        -------
        str
            The label for the drive
        """
        label = f"{self._name}_drive"
        return label


class Layer:
    """
    A class to represent a layer of quantum operations applied simultaneously in a quantum circuit.
    """

    def __init__(self, *operations: Operation) -> None:
        """
        __init__ Initializes the Layer object.

        Parameters
        ----------
        *operations : Operation
            The operations that make up the layer.

        Raises
        ------
        ValueError
            If the layer does not contain any operations.
        ValueError
            If any operation is not an Operation object.
        ValueError
            If any operation acts on the same qubit as another operation.
        ValueError
            If any operation has a different time than the others.
        """
        self._ops = list(operations)

        if len(self._ops) == 0:
            raise ValueError("Layer must contain at least one operation")

        for op in self._ops:
            if not isinstance(op, Operation):
                raise ValueError("Each operation must be an Operation object")

        qubits = list(chain.from_iterable(op.qubits for op in self._ops))
        qubit_set = set(qubits)

        if len(qubit_set) != len(qubits):
            raise ValueError("Operations must act on unique qubits within each layer")

        self._qubits = qubits

        times = tuple(op.time for op in operations)
        if not all_equal(times):
            raise ValueError("All operations must have the same time")
        self._time = next(iter(times))

        max_duration = max(op.duration for op in operations)
        for op in self._ops:
            if op.duration < max_duration:
                qubit_str = ", ".join(op.qubits)
                warnings.warn(
                    f"Operation {op.name} acting on {(qubit_str)} has a smaller duration than the layer, {max_duration}. Setting duration of operation to layer duration."
                )
                op.duration = max_duration
        self._duration = max_duration

    def __eq__(self, other) -> bool:
        """
        __eq__ Checks if the Layer object is equal to another object.

        Parameters
        ----------
        other : _type_
            The object to compare to.

        Returns
        -------
        bool
            True if the Layer object is equal to the other object, False otherwise.
        """
        if isinstance(other, Layer):
            if self.time != other.time:
                return False

            if self._qubits != other._qubits:
                return False

            if self.duration != other.duration:
                return False

            sorted_ops = sorted(self._ops, key=lambda op: op.name)
            other_ops = sorted(other._ops, key=lambda op: op.name)

            for op, other_op in zip(sorted_ops, other_ops):
                if op != other_op:
                    return False

            return True
        return False

    def __ne__(self, other) -> bool:
        """
        __ne__ Checks if the Layer object is not equal to another object.

        Parameters
        ----------
        other : _type_
            The object to compare to.

        Returns
        -------
        bool
            True if the Layer object is not equal to the other object, False otherwise.
        """
        return not self.__eq__(other)

    def __copy__(self) -> Self:
        """
        __copy__ Returns a shallow copy of the Layer object.

        Returns
        -------
        Self
            A shallow copy of the Layer object.
        """
        layer_copy = self.__class__(*self._ops)
        layer_copy.time = self.time
        layer_copy.duration = self.duration
        return layer_copy

    def __str__(self) -> str:
        """
        __str__ Returns a string of the Layer object.

        Returns
        -------
        str
            A string of the Layer object.
        """
        op_str_iter = (str(op) for op in self._ops)
        layer_repr_str = ",".join(op_str_iter)
        return layer_repr_str

    def __repr__(self) -> str:
        """
        __repr__ Returns a string representation of the Layer object.

        Returns
        -------
        str
            The string representation of the Layer object.
        """
        op_str = repr(self._ops)
        repr_str = f"Layer(operations={op_str})"
        return repr_str

    def __iter__(self) -> Iterator[Operation]:
        """
        __iter__ Returns an iterator over the operations in the layer.

        Returns
        -------
        Iterator[Operation]
            An iterator over the operations in the layer.
        """
        return iter(self._ops)

    def __getitem__(self, index: int) -> Operation:
        """
        __getitem__ Returns the operation at the given index.

        Parameters
        ----------
        index : int
            The index of the operation to return.

        Returns
        -------
        Operation
            The operation at the given index.
        """
        return self._ops[index]

    def __delitem__(self, index: int) -> None:
        """
        __delitem__ Deletes the operation at the given index.

        Parameters
        ----------
        index : int
            The index of the operation to delete.
        """
        del self._ops[index]

    def __reversed__(self) -> Iterator[Operation]:
        """
        __reversed__ Returns an iterator over the operations in the layer in reverse order.

        Returns
        -------
        Iterator[Operation]
            An iterator over the operations in the layer in reverse order.

        Yields
        ------
        Iterator[Operation]
            The operations in the layer in reverse order.
        """
        return reversed(self._ops)

    def __contains__(self, item) -> bool:
        """
        __contains__ Checks if the layer contains the given item.

        Parameters
        ----------
        item : _type_
            The item to check for in the layer.

        Returns
        -------
        bool
            True if the item is in the layer, False otherwise.
        """
        return item in self._ops

    def __len__(self) -> int:
        """
        __len__ Returns the number of operations in the layer.

        Returns
        -------
        int
            The number of operations in the layer.
        """
        return len(self._ops)

    def __add__(self, other: Operation | Self) -> Self | Circuit:
        """
        __add__ Adds another object to the Layer object.

        Parameters
        ----------
        other : Operation | Layer
            The object to add to the Layer object.

        Returns
        -------
        Self
            The Layer object with the added object.

        Raises
        ------
        NotImplementedError
            If the other object is a Layer object
        ValueError
            If the other object is not an Operation or Layer object
        """
        if isinstance(other, Operation):
            try:
                ops = (*self._ops, other)
                layer = self.__class__(*ops)
            except ValueError as err:
                raise ValueError("Cannot add the operation to the layer") from err
            return layer

        if isinstance(other, self.__class__):
            if self.time == other.time:
                try:
                    ops = (*self._ops, *other.operations)
                    return self.__class__(*ops)
                except ValueError as err:
                    raise ValueError("Cannot add the two layers together") from err
            else:
                if self.time < other.time:
                    layers = (self, other)
                else:
                    layers = (other, self)

                try:
                    circuit = Circuit(*layers)
                except ValueError as err:
                    raise ValueError(
                        "Cannot add the join the two layer to a new circuit"
                    ) from err

                return circuit

        raise TypeError("Can only add Operations or Layer to Layer")

    def __radd__(self, other: Operation | Self) -> Self | Circuit:
        """
        __radd__ Adds another object to the Layer object.

        Parameters
        ----------
        other : Operation | Layer
            The object to add to the Layer object.

        Returns
        -------
        Self
            The Layer object with the added object.

        Raises
        ------
        NotImplementedError
            If the other object is a Layer object
        ValueError
            If the other object is not an Operation or Layer object
        """
        if isinstance(other, Operation):
            try:
                ops = (other, *self._ops)
                layer = self.__class__(*ops)
            except ValueError as err:
                raise ValueError("Cannot add the operation to the layer") from err
            return layer

        if isinstance(other, self.__class__):
            if self.time == other.time:
                try:
                    ops = (*other.operations, *self._ops)
                    return self.__class__(*ops)
                except ValueError as err:
                    raise ValueError("Cannot add the two layers together") from err
            else:
                if self.time < other.time:
                    layers = (self, other)
                else:
                    layers = (other, self)

                try:
                    circuit = Circuit(*layers)
                except ValueError as err:
                    raise ValueError(
                        "Cannot add the join the two layer to a new circuit"
                    ) from err

                return circuit

        raise TypeError("Can only add Operations or Layer to Layer")

    def __iadd__(self, other: Operation | Self) -> Self:
        """
        __iadd__ Inplace adds another Operation or Layer to the Layer object.

        Parameters
        ----------
        other : Operation | Layer
            The Operation or Layer to add to the Layer object.

        Returns
        -------
        Self
            The Layer object with the added operation.

        Raises
        ------
        ValueError
            If operation is not an Operation object
        """
        if isinstance(other, Operation):
            try:
                self.append(other)
            except ValueError as err:
                raise ValueError("Cannot add the operation to the layer") from err
            return self

        if isinstance(other, self.__class__):
            try:
                self.merge(other)
            except ValueError as err:
                raise ValueError("Cannot merge the layer with the layer") from err
            return self

        raise TypeError("Can only add Operation objects")

    @property
    def operations(self) -> List[Operation]:
        """
        operations Returns the operations in the layer.

        Returns
        -------
        List[Operation]
            The operations in the layer.
        """
        return self._ops

    @property
    def num_operations(self) -> int:
        """
        num_operations Returns the number of operations in the layer.

        Returns
        -------
        int
            The number of operations in the layer.
        """
        return len(self._ops)

    @property
    def qubits(self) -> List[str]:
        """
        qubits Returns the qubits that the operations in the layer act on.

        Returns
        -------
        List[str]
            The qubits that the operations in the layer act on.
        """
        return self._qubits

    @property
    def num_qubits(self) -> int:
        """
        num_qubits Returns the number of qubits that the operations in the layer act on.

        Returns
        -------
        int
            The number of qubits that the operations in the layer act on.
        """
        return len(self._qubits)

    @property
    def time(self) -> float:
        """
        time Returns the starting time at which the operations in the layer are applied.

        Returns
        -------
        float
            The starting time at which the operations in the layer are applied.
        """
        return self._time

    @time.setter
    def time(self, new_time: float) -> None:
        """
        time Sets the starting time at which the operations in the layer are applied.

        Parameters
        ----------
        new_time : float
            The new starting time.

        Raises
        ------
        ValueError
            If new_time is not a float.
        ValueError
            If new_time is less than 0.
        """
        if not isinstance(new_time, float):
            raise ValueError("new_time must be a float")

        if new_time < 0:
            raise ValueError("new_time must be greater than or equal to 0")

        self._time = new_time

        for op in self._ops:
            op.time = new_time

    @property
    def duration(self) -> float:
        """
        duration Returns the duration of the layer.

        Returns
        -------
        float
            The duration of the layer.
        """
        return self._duration

    @duration.setter
    def duration(self, new_duration: float) -> None:
        """
        duration Sets the duration of the layer.

        Parameters
        ----------
        new_duration : float
            The new duration of the

        Raises
        ------
        ValueError
            If new_duration is not a float.
        ValueError
            If new_duration is less than or equal to 0.
        """
        if not isinstance(new_duration, float):
            raise ValueError("new_duration must be a float")

        if new_duration <= 0:
            raise ValueError("new_duration must be greater than 0")

        if new_duration < self.duration:
            warnings.warn("Layer duration is smaller than new duration.")

        self._duration = new_duration

        for op in self._ops:
            op.duration = new_duration

    def append(self, operation: Operation) -> None:
        """
        append Appends an Operation object to the Layer object.

        Parameters
        ----------
        operation : Operation
            The Operation object to append to the Layer object.

        Raises
        ------
        ValueError
            If operation is not an Operation object
        ValueError
            If the operation time does not match the Layer time
        ValueError
            If the operation qubits overlap with the Layer qubits
        """
        if not isinstance(operation, Operation):
            raise TypeError("Can only append Operation objects")

        if operation.time != self.time:
            raise ValueError("Operation time must match layer time")

        qubit_set = set(self.qubits)
        overlap_qubits = qubit_set.intersection(operation.qubits)

        if overlap_qubits:
            raise ValueError(
                f"Operation {operation.name} qubits overlap with layer qubits"
            )

        if operation.duration <= self.duration:
            warnings.warn(
                f"Operation {operation.name} has a smaller duration than the layer, {self.duration}. Setting duration of operation to layer duration."
            )
            operation.duration = self.duration
        else:
            warnings.warn(
                f"Layer duration is smaller than operation {operation.name} duration. Setting layer duration to operation duration."
            )
            self.duration = operation.duration
            for op in self._ops:
                op.duration = operation.duration

        self._ops.append(operation)
        self._qubits.extend(operation.qubits)

    def merge(self, layer: Self) -> None:
        """
        merge Merges another Layer object with the Layer object.

        Parameters
        ----------
        layer : Self
            The Layer object to merge with the Layer object.

        Raises
        ------
        ValueError
            If layer is not a Layer object
        ValueError
            If the layer time does not match the Layer time
        ValueError
            If the layer qubits overlap with the Layer qubits
        """
        if not isinstance(layer, self.__class__):
            raise TypeError("Can only merge Layer objects")

        if self.time != layer.time:
            raise ValueError("Layer time must match layer time")

        qubit_set = set(self.qubits)
        overlap_qubits = qubit_set.intersection(layer.qubits)
        if overlap_qubits:
            raise ValueError("Layer qubits overlap with layer qubits")

        if self.duration != layer.duration:
            warnings.warn(
                "Layer durations do not match. Setting duration of merged layer to the maximum duration."
            )
            max_duration = max(self.duration, layer.duration)
            if self.duration < max_duration:
                self.duration = max_duration
                for op in self._ops:
                    op.duration = max_duration
            else:
                layer.duration = max_duration
                for op in layer.operations:
                    op.duration = max_duration

        self._ops.extend(layer.operations)
        self._qubits.extend(layer.qubits)


class Circuit:
    """
    A class to represent a quantum circuit, implemented by a sequence of layers.
    """

    def __init__(self, *layers: Layer) -> None:
        self._layers = list(layers)
        for layer in self._layers:
            if not isinstance(layer, Layer):
                raise ValueError("Each layer must be a Layer object")

        prev_time = None
        for ind, layer in enumerate(self._layers):
            if prev_time is None:
                prev_time = layer.time + layer.duration
            else:
                if layer.time < prev_time:
                    raise ValueError(
                        f"Layers must be in chronological order. Layer at index {ind} starts at time {layer.time}, before the previous layer has ended (at {prev_time})."
                    )

                if layer.time > prev_time:
                    warnings.warn(
                        f"Layer at index {ind} starts at time {layer.time}, while the previous layer has ended (at {prev_time}). Extending the duration of the previous layer to match the start time of the current layer."
                    )

                    prev_layer = self._layers[ind - 1]
                    prev_layer.duration = layer.time - prev_layer.time

        qubit_set = set(chain.from_iterable(layer.qubits for layer in self._layers))
        self._qubits = list(qubit_set)

        try:
            first_layer = next(iter(self._layers))
            self._time = first_layer.time

        except StopIteration:
            self._time = 0.0

        if self._layers:
            self._duration = sum(layer.duration for layer in self._layers)
        else:
            self._duration = None

    def __eq__(self, other: Self) -> bool:
        """
        __eq__ Checks if the circuit is equal to another circuit.

        Parameters
        ----------
        other : Self
            The circuit to compare to.

        Returns
        -------
        bool
            True if the circuits are equal, False otherwise
        """
        if not isinstance(other, self.__class__):
            return False

        if self.time != other.time:
            return False

        if self.duration != other.duration:
            return False

        return self._layers == other._layers

    def __ne__(self, other: Self) -> bool:
        """
        __ne__ Checks if the circuit is not equal to another circuit.

        Parameters
        ----------
        other : Self
            The circuit to compare to.

        Returns
        -------
        bool
            True if the circuits are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __str__(self) -> str:
        """
        __str__ Returns a readable string representation of the circuit.

        Returns
        -------
        str
            A string representation of the circuit.
        """
        layer_strs = []
        for ind, layer in enumerate(self._layers):
            start_time = layer.time
            end_time = start_time + layer.duration
            layer_strs.append(
                f"Layer {ind} from {start_time} to {end_time}:\n\t{layer!s}"
            )

        circ_str = "\n".join(layer_strs)
        return circ_str

    def __repr__(self) -> str:
        """
        __repr__ Returns a unique string representation of the circuit.

        Returns
        -------
        str
            A string representation of the circuit.
        """
        layer_str = repr(self._layers)
        repr_str = f"Circuit(layers={layer_str})"
        return repr_str

    def __len__(self) -> int:
        """
        __len__ Returns the number of layers in the circuit.

        Returns
        -------
        int
            The number of layers in the circuit.
        """
        return len(self._layers)

    def __iter__(self) -> Iterator[Layer]:
        """
        __iter__ Returns an iterator over the layers of the circuit.

        Returns
        -------
        Iterator[Layer]
            An iterator over the layers of the circuit.
        """
        return iter(self._layers)

    def __getitem__(self, index: int) -> Layer:
        """
        __getitem__ Returns the layer at the given index.

        Parameters
        ----------
        index : int
            The index of the layer to return.

        Returns
        -------
        Layer
            The layer at the given index.
        """
        return self._layers[index]

    def __contains__(self, layer: Layer) -> bool:
        """
        __contains__ Checks if the circuit contains the given layer.

        Parameters
        ----------
        layer: Layer
            The layer to check for.

        Returns
        -------
        bool
            True if the circuit contains the layer, False otherwise.
        """
        return layer in self._layers

    def __reverse__(self) -> Iterator[Layer]:
        """
        __reverse__ Returns an iterator over the layers of the circuit in reverse order.

        Returns
        -------
        Iterator[Layer]
            An iterator over the layers of the circuit in reverse order.
        """
        return reversed(self._layers)

    def __del___(self, index: int) -> None:
        """
        __del__ Deletes the layer at the given index.

        Parameters
        ----------
        index : int
            The index of the layer to delete.
        """
        del self._layers[index]

    def __copy__(self) -> Self:
        """
        __copy__ Returns a shallow copy of the circuit.

        Returns
        -------
        Self
            A shallow copy of the circuit.
        """
        circuit_copy = self.__class__(*self._layers)
        return circuit_copy

    @property
    def layers(self) -> List[Layer]:
        """
        layers Returns the layers of the circuit.

        Returns
        -------
        List[Layer]
            The layers of the circuit.
        """
        return self._layers

    @property
    def num_layers(self) -> int:
        """
        num_layers Returns the number of layers in the circuit.

        Returns
        -------
        int
            The number of layers in the circuit.
        """
        return len(self._layers)

    @property
    def qubits(self) -> List[str]:
        """
        qubits Returns the qubits on which the circuit operates.

        Returns
        -------
        List[str]
            The qubits on which the circuit operates.
        """
        return self._qubits

    @property
    def num_qubits(self) -> int:
        """
        num_qubits Returns the number of qubits on which the circuit operates.

        Returns
        -------
        int
            The number of qubits on which the circuit operates.
        """
        return len(self._qubits)

    @property
    def time(self) -> float:
        """
        time Returns the starting time of the circuit.

        Returns
        -------
        float
            The starting time of the circuit.
        """
        return self._time

    @time.setter
    def time(self, new_time: float) -> None:
        """
        time Sets the starting time of the circuit.

        Parameters
        ----------
        new_time : float
            The new starting time of the circuit.

        Raises
        ------
        ValueError
            If new_time is not a float.
        ValueError
            If new_time is less than 0.
        """
        if not isinstance(new_time, float):
            raise ValueError("new_time must be a float")

        if new_time < 0:
            raise ValueError("time must be greater than or equal to 0")

        self._time = new_time

        time = new_time
        for layer in self._layers:
            layer.time = time
            time += layer.duration

    @property
    def duration(self) -> float | None:
        """
        duration Returns the duration of the circuit.

        Returns
        -------
        float | None
            The duration of the circuit. If the circuit has no layers, returns None.
        """
        return self._duration

    def append(self, layer: Layer) -> None:
        """
        append Appends a layer to the circuit.

        Parameters
        ----------
        layer : Layer
            The layer to append to the circuit.

        Raises
        ------
        ValueError
            If layer is not a Layer object.
        ValueError
            If layer time is less than circuit time.
        ValueError
            If layer duration exceeds circuit duration.
        """
        if not isinstance(layer, Layer):
            raise ValueError("Can only append Layer objects")

        if layer.time < self.time:
            raise ValueError("Layer time must be greater than or equal to circuit time")

        if self.duration is not None:
            if layer.time + layer.duration > self.duration:
                raise ValueError("Layer duration exceeds circuit duration")

        self._layers.append(layer)
        self._qubits.extend(layer.qubits)
