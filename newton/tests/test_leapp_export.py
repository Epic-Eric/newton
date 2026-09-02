# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check LEAPP PT2 exports for several Newton controller families.

Run from the Newton repository root with::

    uv run --no-sync -m newton.tests -k test_leapp_export

Every case reuses one controller instance for LEAPP's discovery and APIC
capture passes, uses different values in those passes, and validates the
exported runtime on additional input sequences.
"""

from __future__ import annotations

import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.actuators import ControllerPD, ControllerPID
from newton.controllers import ControllerJointImpedance, ControllerJointImpedanceModelFree

try:
    import leapp
    import torch
    from leapp import InferenceManager, annotate
except ImportError:
    leapp = None
    torch = None
    InferenceManager = None
    annotate = None


GRAPH_NAME = "newton_controller"
ATOL = 1.0e-5

SourceValues = dict[str, np.ndarray]
SourceSequence = tuple[SourceValues, ...]


def _f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _warp_arrays(values: SourceValues, device: wp.DeviceLike) -> dict[str, wp.array[Any]]:
    return {name: wp.array(value, dtype=wp.float32, device=device) for name, value in values.items()}


def _annotate_inputs(node_name: str, arrays: dict[str, wp.array[Any]]) -> dict[str, wp.array[Any]]:
    traced = annotate.input_tensors(node_name, arrays)
    if len(arrays) == 1:
        traced = (traced,)
    return dict(zip(arrays, traced, strict=True))


def _joint_values(q, qd, q_des, qd_des, **extras) -> SourceValues:
    values = {
        "joint_q": _f32(q),
        "joint_qd": _f32(qd),
        "joint_q_des": _f32(q_des),
        "joint_qd_des": _f32(qd_des),
    }
    values.update({name: _f32(value) for name, value in extras.items()})
    return values


def _actuator_values(positions, velocities, target_pos, target_vel, feedforward) -> SourceValues:
    return {
        "positions": _f32(positions),
        "velocities": _f32(velocities),
        "target_pos": _f32(target_pos),
        "target_vel": _f32(target_vel),
        "feedforward": _f32(feedforward),
    }


def _build_arm(device: wp.DeviceLike, link_count: int = 2) -> newton.Model:
    builder = newton.ModelBuilder()
    inertia = wp.mat33(np.diag([0.02, 0.02, 0.02]).astype(np.float32))
    parent = -1

    for index in range(link_count):
        link = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.25, 0.0, 0.0),
            inertia=inertia,
            lock_inertia=True,
        )
        builder.add_joint_revolute(
            parent=parent,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.5 if index else 0.0, 0.0, 0.0)),
            label=f"joint_{index}",
        )
        parent = link

    builder.add_articulation(list(range(link_count)), label="arm")
    return builder.finalize(device=device)


class ControllerExportCase(ABC):
    """Shared LEAPP export and runtime lifecycle for one controller family."""

    node_name: str
    stateful = False
    dt = 0.01

    def __init__(self, device: wp.DeviceLike):
        self.device = wp.get_device(device)

    @property
    @abstractmethod
    def pass_values(self) -> SourceSequence:
        """Different values for discovery and capture, with matching structure."""

    @property
    @abstractmethod
    def runtime_values(self) -> SourceSequence:
        """Values used only after export."""

    @abstractmethod
    def make_controller(self) -> Any:
        """Construct one controller instance."""

    def make_capture_context(self, controller: Any) -> Any:
        """Allocate non-input resources before LEAPP capture starts."""
        return None

    @abstractmethod
    def capture_step(
        self,
        controller: Any,
        inputs: dict[str, wp.array[Any]],
        context: Any,
        pass_index: int,
    ) -> wp.array[Any]:
        """Capture one controller invocation and return its output array."""

    @abstractmethod
    def run_native(self, values: SourceSequence) -> np.ndarray:
        """Run a sequence through one native controller instance."""

    def export(self, output_root: Path) -> Path:
        """Export one controller reused across discovery and capture."""
        controller = self.make_controller()
        context = self.make_capture_context(controller)

        leapp.start(name=GRAPH_NAME, save_path=str(output_root))
        try:
            for pass_index, values in enumerate(self.pass_values):
                traced_inputs = _annotate_inputs(
                    self.node_name,
                    _warp_arrays(values, self.device),
                )
                output = self.capture_step(controller, traced_inputs, context, pass_index)
                annotate.output_tensors(
                    self.node_name,
                    {"joint_f": output},
                    export_with="pt2",
                )
        finally:
            leapp.stop()

        leapp.compile_graph(
            visualize=False,
            validate=True,
            atol=ATOL,
            strict=True,
        )
        return output_root / GRAPH_NAME / f"{GRAPH_NAME}.yaml"

    def run_exported(self, graph_path: Path, values: SourceSequence) -> np.ndarray:
        """Run a sequence through one runtime, preserving exported state."""
        manager = InferenceManager(str(graph_path))
        runtime_device = manager.nodes[self.node_name].device
        outputs = []

        for source_values in values:
            runtime_inputs = {
                f"{self.node_name}/{name}": torch.as_tensor(value, device=runtime_device)
                for name, value in source_values.items()
            }
            runtime_outputs = manager.run_policy(runtime_inputs)
            outputs.append(runtime_outputs[f"{self.node_name}/joint_f"].detach().cpu().numpy())

        return np.stack(outputs)


class ModelBasedJointImpedanceCase(ControllerExportCase):
    """Model-based controller with internally computed dynamics terms."""

    node_name = "model_based_joint_impedance"

    @property
    def pass_values(self) -> SourceSequence:
        return (
            _joint_values([0.3, -0.4], [0.1, 0.2], [0.0, 0.5], [0.0, 0.0]),
            _joint_values([-0.2, 0.6], [-0.3, 0.4], [0.7, -0.1], [0.2, -0.2]),
        )

    @property
    def runtime_values(self) -> SourceSequence:
        first = _joint_values([0.15, -0.25], [-0.05, 0.3], [0.4, 0.1], [0.1, -0.1])
        middle = _joint_values([-0.45, 0.35], [0.25, -0.2], [-0.1, 0.6], [0.0, 0.25])
        return (first, middle, first)

    def make_controller(self) -> ControllerJointImpedance:
        model = _build_arm(self.device)
        return ControllerJointImpedance(
            model,
            stiffness=wp.array([50.0, 30.0], dtype=wp.float32, device=self.device),
            damping=wp.array([5.0, 3.0], dtype=wp.float32, device=self.device),
        )

    @staticmethod
    def _prepare_step(controller, arrays):
        inputs = controller.input()
        outputs = controller.output()
        for name, value in arrays.items():
            setattr(inputs, name, value)
        return inputs, outputs

    def capture_step(self, controller, inputs, context, pass_index) -> wp.array[Any]:
        controller_inputs, outputs = self._prepare_step(controller, inputs)
        with annotate.warp_op(self.node_name):
            controller.step(inputs=controller_inputs, outputs=outputs, dt=self.dt)
        return outputs.joint_f

    def run_native(self, values: SourceSequence) -> np.ndarray:
        controller = self.make_controller()
        outputs = []
        for source_values in values:
            inputs, result = self._prepare_step(
                controller,
                _warp_arrays(source_values, self.device),
            )
            controller.step(inputs=inputs, outputs=result, dt=self.dt)
            wp.synchronize_device(self.device)
            outputs.append(result.joint_f.numpy().copy())
        return np.stack(outputs)


class ModelFreeJointImpedanceCase(ControllerExportCase):
    """Heterogeneous model-free controller with live gains and dynamics inputs."""

    node_name = "model_free_joint_impedance"

    @staticmethod
    def _values(*, q, qd, q_des, qd_des, qdd, gravity, coriolis, mass_matrix, stiffness, damping):
        return _joint_values(
            q,
            qd,
            q_des,
            qd_des,
            joint_qdd=qdd,
            gravity_force=gravity,
            coriolis_force=coriolis,
            mass_matrix=mass_matrix,
            stiffness=stiffness,
            damping=damping,
        )

    @property
    def pass_values(self) -> SourceSequence:
        return (
            self._values(
                q=[0.3, -0.4, 0.2],
                qd=[0.1, 0.2, -0.1],
                q_des=[0.0, 0.5, 0.7],
                qd_des=[0.0, 0.0, 0.2],
                qdd=[0.2, -0.1, 0.3],
                gravity=[1.0, -2.0, 0.5],
                coriolis=[0.1, 0.2, -0.3],
                mass_matrix=[[[2.0, 0.1], [0.1, 1.5]], [[1.2, 91.0], [-73.0, 44.0]]],
                stiffness=[50.0, 30.0, 20.0],
                damping=[5.0, 3.0, 2.0],
            ),
            self._values(
                q=[-0.2, 0.6, -0.5],
                qd=[-0.3, 0.4, 0.2],
                q_des=[0.7, -0.1, 0.25],
                qd_des=[0.2, -0.2, -0.1],
                qdd=[-0.4, 0.3, 0.1],
                gravity=[-1.0, 1.5, -0.75],
                coriolis=[0.3, -0.2, 0.4],
                mass_matrix=[[[1.2, -0.2], [-0.2, 2.0]], [[0.8, -55.0], [32.0, 17.0]]],
                stiffness=[12.0, 45.0, 27.0],
                damping=[1.0, 6.0, 2.5],
            ),
        )

    @property
    def runtime_values(self) -> SourceSequence:
        first = self._values(
            q=[0.15, -0.25, 0.4],
            qd=[-0.05, 0.3, -0.2],
            q_des=[0.4, 0.1, -0.1],
            qd_des=[0.1, -0.1, 0.25],
            qdd=[0.05, 0.2, -0.15],
            gravity=[0.75, -0.25, 0.6],
            coriolis=[-0.1, 0.4, -0.2],
            mass_matrix=[[[1.5, 0.25], [0.25, 1.8]], [[0.9, 99.0], [-99.0, 123.0]]],
            stiffness=[40.0, 25.0, 15.0],
            damping=[4.0, 2.0, 3.0],
        )
        unused_padding_changed = {
            **first,
            "mass_matrix": _f32([[[1.5, 0.25], [0.25, 1.8]], [[0.9, -700.0], [800.0, -900.0]]]),
        }
        middle = self._values(
            q=[-0.2, 0.45, -0.35],
            qd=[0.2, -0.15, 0.3],
            q_des=[0.1, -0.3, 0.55],
            qd_des=[-0.1, 0.2, -0.25],
            qdd=[-0.2, 0.1, 0.4],
            gravity=[-0.5, 1.25, -0.8],
            coriolis=[0.2, -0.35, 0.15],
            mass_matrix=[[[2.1, -0.4], [-0.4, 1.3]], [[1.1, 12.0], [13.0, 14.0]]],
            stiffness=[18.0, 35.0, 22.0],
            damping=[2.0, 5.0, 1.5],
        )
        return (first, unused_padding_changed, middle, first)

    def make_controller(self) -> ControllerJointImpedanceModelFree:
        return ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=wp.array([2, 1], dtype=wp.int32, device=self.device),
            stiffness=None,
            damping=None,
            use_gravity_compensation=True,
            use_coriolis_compensation=True,
            use_inertia_decoupling=True,
            has_qdd_feedforward=True,
            device=self.device,
        )

    @staticmethod
    def _prepare_step(controller, arrays):
        inputs = controller.input()
        outputs = controller.output()
        for name, value in arrays.items():
            setattr(inputs, name, value)
        return inputs, outputs

    def capture_step(self, controller, inputs, context, pass_index) -> wp.array[Any]:
        controller_inputs, outputs = self._prepare_step(controller, inputs)
        with annotate.warp_op(self.node_name):
            controller.step(inputs=controller_inputs, outputs=outputs, dt=self.dt)
        return outputs.joint_f

    def run_native(self, values: SourceSequence) -> np.ndarray:
        controller = self.make_controller()
        outputs = []
        for source_values in values:
            inputs, result = self._prepare_step(
                controller,
                _warp_arrays(source_values, self.device),
            )
            controller.step(inputs=inputs, outputs=result, dt=self.dt)
            wp.synchronize_device(self.device)
            outputs.append(result.joint_f.numpy().copy())
        return np.stack(outputs)


class ActuatorControllerCase(ControllerExportCase):
    """Shared one-to-one input mapping for actuator controller cases."""

    actuator_count = 3

    def _identity_indices(self) -> wp.array[wp.uint32]:
        return wp.array([0, 1, 2], dtype=wp.uint32, device=self.device)

    def make_capture_context(self, controller):
        return self._identity_indices()

    def _compute(self, controller, inputs, indices, forces, state) -> None:
        controller.compute(
            positions=inputs["positions"],
            velocities=inputs["velocities"],
            target_pos=inputs["target_pos"],
            target_vel=inputs["target_vel"],
            feedforward=inputs["feedforward"],
            pos_indices=indices,
            vel_indices=indices,
            target_pos_indices=indices,
            target_vel_indices=indices,
            forces=forces,
            state=state,
            dt=self.dt,
            device=self.device,
        )


class PDControllerCase(ActuatorControllerCase):
    """Stateless PD controller over three directly aligned actuators."""

    node_name = "pd_controller"

    @property
    def pass_values(self) -> SourceSequence:
        return (
            _actuator_values(
                [0.3, -0.4, 0.2],
                [0.1, 0.2, -0.1],
                [0.0, 0.5, 0.7],
                [0.0, 0.0, 0.2],
                [0.2, -0.1, 0.3],
            ),
            _actuator_values(
                [-0.2, 0.6, -0.5],
                [-0.3, 0.4, 0.2],
                [0.7, -0.1, 0.25],
                [0.2, -0.2, -0.1],
                [-0.4, 0.3, 0.1],
            ),
        )

    @property
    def runtime_values(self) -> SourceSequence:
        first = _actuator_values(
            [0.15, -0.25, 0.4],
            [-0.05, 0.3, -0.2],
            [0.4, 0.1, -0.1],
            [0.1, -0.1, 0.25],
            [0.05, 0.2, -0.15],
        )
        middle = _actuator_values(
            [-0.4, 0.2, 0.1],
            [0.25, -0.35, 0.45],
            [0.6, -0.2, 0.3],
            [-0.3, 0.4, -0.2],
            [0.2, -0.1, 0.4],
        )
        return (first, middle, first)

    def make_controller(self) -> ControllerPD:
        return ControllerPD(
            kp=wp.array([50.0, 30.0, 20.0], dtype=wp.float32, device=self.device),
            kd=wp.array([5.0, 3.0, 2.0], dtype=wp.float32, device=self.device),
            const_effort=wp.array([0.25, -0.5, 0.75], dtype=wp.float32, device=self.device),
        )

    def capture_step(self, controller, inputs, context, pass_index) -> wp.array[Any]:
        forces = wp.zeros(self.actuator_count, dtype=wp.float32, device=self.device)
        with annotate.warp_op(self.node_name):
            self._compute(controller, inputs, context, forces, None)
        return forces

    def run_native(self, values: SourceSequence) -> np.ndarray:
        controller = self.make_controller()
        indices = self._identity_indices()
        outputs = []
        for source_values in values:
            forces = wp.zeros(self.actuator_count, dtype=wp.float32, device=self.device)
            self._compute(controller, _warp_arrays(source_values, self.device), indices, forces, None)
            wp.synchronize_device(self.device)
            outputs.append(forces.numpy().copy())
        return np.stack(outputs)


class PIDControllerCase(ActuatorControllerCase):
    """Stateful PID controller exercised through accumulation and clamping."""

    node_name = "pid_controller"
    stateful = True
    dt = 0.5

    @property
    def pass_values(self) -> SourceSequence:
        return (
            _actuator_values(
                [0.0] * 3,
                [0.1, -0.2, 0.3],
                [1.5, -1.5, 1.5],
                [0.0] * 3,
                [0.2, -0.1, 0.3],
            ),
            _actuator_values(
                [0.2, -0.3, 0.4],
                [-0.1, 0.4, -0.2],
                [-0.7, 0.8, -1.2],
                [0.2, -0.2, 0.1],
                [-0.4, 0.5, -0.2],
            ),
        )

    @property
    def runtime_values(self) -> SourceSequence:
        repeated = _actuator_values(
            [0.0] * 3,
            [0.1, -0.2, 0.3],
            [1.5, -1.5, 1.5],
            [0.0] * 3,
            [0.2, -0.1, 0.3],
        )
        reverse = _actuator_values(
            [0.0] * 3,
            [-0.2, 0.1, -0.4],
            [-1.5, 1.5, -1.5],
            [0.1, -0.1, 0.2],
            [-0.3, 0.4, -0.1],
        )
        return (repeated, repeated, repeated, reverse)

    def make_controller(self) -> ControllerPID:
        controller = ControllerPID(
            kp=wp.array([2.0, 3.0, 4.0], dtype=wp.float32, device=self.device),
            ki=wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=self.device),
            kd=wp.array([0.5, 0.25, 0.75], dtype=wp.float32, device=self.device),
            integral_max=wp.array([1.0, 1.0, 1.0], dtype=wp.float32, device=self.device),
            const_effort=wp.array([0.25, -0.5, 0.75], dtype=wp.float32, device=self.device),
        )
        controller.finalize(self.device, self.actuator_count)
        return controller

    def make_capture_context(self, controller):
        return {
            "indices": self._identity_indices(),
            "states": tuple(controller.state(self.actuator_count, self.device) for _ in self.pass_values),
        }

    def capture_step(self, controller, inputs, context, pass_index) -> wp.array[Any]:
        current_state = ControllerPID.State(
            integral=annotate.state_tensors(
                self.node_name,
                {"integral": context["states"][pass_index].integral},
            )
        )
        next_state = controller.state(self.actuator_count, self.device)
        forces = wp.zeros(self.actuator_count, dtype=wp.float32, device=self.device)

        with annotate.warp_op(self.node_name):
            self._compute(controller, inputs, context["indices"], forces, current_state)
            controller.update_state(current_state, next_state)

        annotate.update_state(self.node_name, {"integral": next_state.integral})
        return forces

    def run_native(self, values: SourceSequence) -> np.ndarray:
        controller = self.make_controller()
        indices = self._identity_indices()
        current_state = controller.state(self.actuator_count, self.device)
        next_state = controller.state(self.actuator_count, self.device)
        outputs = []

        for source_values in values:
            forces = wp.zeros(self.actuator_count, dtype=wp.float32, device=self.device)
            self._compute(
                controller,
                _warp_arrays(source_values, self.device),
                indices,
                forces,
                current_state,
            )
            controller.update_state(current_state, next_state)
            wp.synchronize_device(self.device)
            outputs.append(forces.numpy().copy())
            current_state, next_state = next_state, current_state

        return np.stack(outputs)


@unittest.skipUnless(leapp is not None and wp.is_cuda_available(), "requires LEAPP, PyTorch, and CUDA")
class TestLeappExport(unittest.TestCase):
    def assert_case_round_trip(self, case: ControllerExportCase) -> np.ndarray:
        """Compare one native controller with its exported runtime sequence."""
        native_outputs = case.run_native(case.runtime_values)
        with tempfile.TemporaryDirectory(prefix=f"newton_leapp_{case.node_name}_") as tmp:
            graph_path = case.export(Path(tmp))
            exported_outputs = case.run_exported(graph_path, case.runtime_values)

        np.testing.assert_allclose(exported_outputs, native_outputs, atol=ATOL)
        if not case.stateful:
            np.testing.assert_allclose(exported_outputs[0], exported_outputs[-1], atol=ATOL)
        return exported_outputs

    def test_model_based_joint_impedance(self):
        """Persistent model buffers consume fresh data on every invocation."""
        self.assert_case_round_trip(ModelBasedJointImpedanceCase("cuda"))

    def test_model_free_joint_impedance(self):
        """Live gains, heterogeneous dynamics, and padded matrices round-trip."""
        outputs = self.assert_case_round_trip(ModelFreeJointImpedanceCase("cuda"))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=ATOL)

    def test_pd_controller(self):
        """Stateless PD effort computation survives export."""
        self.assert_case_round_trip(PDControllerCase("cuda"))

    def test_pid_controller_state(self):
        """PID feedback state accumulates, clamps, and persists at runtime."""
        outputs = self.assert_case_round_trip(PIDControllerCase("cuda"))
        self.assertFalse(np.allclose(outputs[0], outputs[1], atol=ATOL))
        np.testing.assert_allclose(outputs[1], outputs[2], atol=ATOL)

    def test_runtime_rejects_wrong_input_shape(self):
        """The PT2 input guard rejects a malformed controller input."""
        case = PDControllerCase("cuda")
        bad_values = dict(case.runtime_values[0])
        bad_values["positions"] = np.zeros(2, dtype=np.float32)

        with tempfile.TemporaryDirectory(prefix="newton_leapp_bad_shape_") as tmp:
            graph_path = case.export(Path(tmp))
            with self.assertRaisesRegex(AssertionError, r"positions.*3"):
                case.run_exported(graph_path, (bad_values,))


if __name__ == "__main__":
    unittest.main()
