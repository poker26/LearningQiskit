"""
Общие функции для экспериментов Aer + IBM Quantum (SamplerV2).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator


def configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_dotenv_from_project(project_root: Path | None = None) -> None:
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / ".env")


def ensure_output_directory(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    output_directory = project_root / "output"
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def save_circuit_figure(
    circuit: QuantumCircuit,
    output_path: Path,
    *,
    style: str = "iqp",
) -> Path:
    figure = circuit.draw(output="mpl", style=style, fold=-1)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    figure.clear()
    return output_path


def run_shot_statistics_on_aer(
    circuit: QuantumCircuit,
    aer_backend: AerSimulator,
    shot_count: int,
) -> dict[str, int]:
    simulation_job = aer_backend.run(circuit, shots=shot_count)
    measurement_result = simulation_job.result()
    return measurement_result.get_counts()


def extract_counts_from_sampler_result(pub_result) -> dict[str, int]:
    register_names = list(pub_result.data)
    if not register_names:
        raise ValueError("В результате Sampler нет классических регистров.")
    if len(register_names) == 1:
        measurement_bit_array = pub_result.data[register_names[0]]
    else:
        measurement_bit_array = pub_result.join_data(register_names)
    return measurement_bit_array.get_counts()


def run_on_ibm_quantum_hardware(
    circuit: QuantumCircuit,
    shot_count: int,
    backend_name: str | None,
    *,
    min_num_qubits: int,
    optimization_level: int = 1,
) -> tuple[dict[str, int], str]:
    if optimization_level not in {0, 1, 2, 3}:
        raise ValueError("optimization_level должен быть 0, 1, 2 или 3.")
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    quantum_channel = "ibm_quantum_platform"
    runtime_service = QiskitRuntimeService(channel=quantum_channel)

    if backend_name:
        selected_backend = runtime_service.backend(backend_name)
    else:
        selected_backend = runtime_service.least_busy(
            min_num_qubits=min_num_qubits,
            operational=True,
            simulator=False,
        )

    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        target=selected_backend.target,
    )
    circuit_for_backend = pass_manager.run(circuit)

    sampler = SamplerV2(mode=selected_backend)
    runtime_job = sampler.run([circuit_for_backend], shots=shot_count)
    print(f"Задание IBM Quantum: job_id={runtime_job.job_id()}, backend={selected_backend.name}")
    primitive_result = runtime_job.result()
    hardware_counts = extract_counts_from_sampler_result(primitive_result[0])
    return hardware_counts, selected_backend.name
