from typing import List

from jax.typing import ArrayLike
from jax import numpy as jnp

import dynamiqs as dq
from dynamiqs.time_array import ConstantTimeArray, TimeArray
from dynamiqs.options import Options
from dynamiqs.result import Result
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver, Tsit5

from .circuits import Circuit, Layer
from ..devices import Device


def get_hamiltonian(device: Device) -> ConstantTimeArray | TimeArray:
    """
    get_hamiltonian Generates the Hamiltonian for the device.

    Parameters
    ----------
    device : Device
        The device for which to generate the Hamiltonian.

    Returns
    -------
    ConstantTimeArray | TimeArray
        The Hamiltonian for the device
    """
    device_hamiltonian = device.get_hamiltonian()
    hamiltonian = dq.constant(device_hamiltonian)

    try:
        drive_terms = device.get_drive_hamiltonian_terms()
        for prefactor, operator in drive_terms:
            drive_operator = dq.modulated(prefactor, operator)
            hamiltonian += drive_operator
    except StopIteration:
        pass

    return hamiltonian


def get_jump_ops(device: Device) -> List[ConstantTimeArray]:
    """
    get_jump_ops Returns the jump operators for the device.

    Parameters
    ----------
    device : Device
        The device for which to generate the jump operators.


    Returns
    -------
    List[ConstantTimeArray | TimeArray]
        The jump operators for the device
    """
    try:
        jump_op_iter = device.get_jump_ops()
        jump_ops = list(map(dq.constant, jump_op_iter))
    except StopIteration:
        null_op = jnp.zeros((device.dim, device.dim))
        jump_ops = [dq.constant(null_op)]
    return jump_ops


def apply_layer(
    device: Device,
    layer: Layer,
    states: ArrayLike,
    num_times: int = 100,
    exp_ops: List[ArrayLike] | None = None,
    gradient: Gradient | None = None,
    options: Options | None = None,
    solver: Solver | None = None,
) -> Result:
    """
    apply_layer Applies a layer of operations to the device.

    Parameters
    ----------
    device : Device
        The device to be simulated
    layer : Layer
        The layer of operations to be simulated.
    states : ArrayLike
        The initial state of the device
    num_times : int, optional
        The number of time steps over which the states and expectation values are saved, by default 100
    exp_ops : List[ArrayLike] | None, optional
        The list of operaetors for which the expectation value is computed, by default None
    gradient : Gradient | None, optional
        The algorithm used to compute the gradient, by default None
    options : Options | None, optional
        The generic options provided to the Lidblad master equation solver, by default None
    solver : Solver | None, optional
        The solver used to perform the integration of the master equation, by default None.
        If None, the default solver used is Tsit5.

    Returns
    -------
    Result
        The result of the simulation.

    Raises
    ------
    ValueError
        If an unknown pulse type is provided.
    """

    end_time = layer.time + layer.duration
    times = jnp.linspace(layer.time, end_time, num_times)

    for qubit in device.qubits:
        qubit.remove_drives()

    for operation in layer:
        label = operation.drive_label
        for qubit_label, pulse, pulse_type in operation:
            match pulse_type:
                case "charge":
                    device[qubit_label].add_charge_drive(label, pulse)
                case "flux":
                    device[qubit_label].add_flux_drive(label, pulse)
                case "detuning":
                    device[qubit_label].add_detuning_drive(label, pulse)
                case _:
                    raise ValueError(f"Unknown pulse type: {pulse_type}")

    hamiltonian = get_hamiltonian(device)
    jump_ops = get_jump_ops(device)

    options = options or Options()
    solver = solver or Tsit5()

    result = dq.mesolve(
        hamiltonian,
        jump_ops,  # type: ignore
        states,
        times,
        gradient=gradient,
        exp_ops=exp_ops,
        options=options,
        solver=solver,
    )

    return result


def apply_circut(
    device: Device,
    circuit: Circuit,
    states: ArrayLike,
    num_times: int = 100,
    exp_ops: List[ArrayLike] | None = None,
    gradient: Gradient | None = None,
    options: Options | None = None,
    solver: Solver | None = None,
) -> List[Result]:
    """
    apply_circut Applies a circuit of layers to the device.

    Parameters
    ----------
    device : Device
        The device to be simulated
    circuit : Circuit
        The circuit of layers to be simulated.
    states : ArrayLike
        The initial state of the device
    num_times : int, optional
        The number of time steps over which the states and expectation values are saved, by default 100
    exp_ops : List[ArrayLike] | None, optional
        The list of operaetors for which the expectation value is computed, by default None
    gradient : Gradient | None, optional
        The algorithm used to compute the gradient, by default None
    options : Options | None, optional
        The generic options provided to the Lidblad master equation solver, by default None
    solver : Solver | None, optional
        The solver used to perform the integration of the master equation, by default None.
        If None, the default solver used is Tsit5.

    Returns
    -------
    Result
        The result of the simulation.

    Raises
    ------
    ValueError
        If an unknown pulse type is provided.
    """
    results = []
    for layer in circuit:
        result = apply_layer(
            device,
            layer,
            states,
            num_times=num_times,
            exp_ops=exp_ops,
            gradient=gradient,
            options=options,
            solver=solver,
        )
        states = result.states[..., -1, :, :]
        results.append(result)

    return results
