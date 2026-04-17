"""
Состояние Белла: симулятор Aer и (опционально) IBM Quantum через SamplerV2.
Сохраняет схему и гистограммы в папку output/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


def build_bell_state_circuit() -> QuantumCircuit:
    bell_circuit = QuantumCircuit(2, 2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_circuit.measure([0, 1], [0, 1])
    bell_circuit.name = "bell_phi_plus"
    return bell_circuit


def ensure_output_directory() -> Path:
    project_root = Path(__file__).resolve().parent
    output_directory = project_root / "output"
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def save_circuit_diagram(bell_circuit: QuantumCircuit, output_directory: Path) -> Path:
    diagram_path = output_directory / "week01_bell_circuit.png"
    figure = bell_circuit.draw(output="mpl", style="iqp", fold=-1)
    figure.savefig(diagram_path, dpi=150, bbox_inches="tight")
    figure.clear()
    return diagram_path


def run_shot_statistics_on_aer(
    bell_circuit: QuantumCircuit,
    aer_backend: AerSimulator,
    shot_count: int,
) -> dict[str, int]:
    simulation_job = aer_backend.run(bell_circuit, shots=shot_count)
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
    bell_circuit: QuantumCircuit,
    shot_count: int,
    backend_name: str | None,
) -> tuple[dict[str, int], str]:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    quantum_channel = "ibm_quantum_platform"
    runtime_service = QiskitRuntimeService(channel=quantum_channel)

    if backend_name:
        selected_backend = runtime_service.backend(backend_name)
    else:
        selected_backend = runtime_service.least_busy(
            min_num_qubits=2,
            operational=True,
            simulator=False,
        )

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        target=selected_backend.target,
    )
    circuit_for_backend = pass_manager.run(bell_circuit)

    sampler = SamplerV2(mode=selected_backend)
    runtime_job = sampler.run([circuit_for_backend], shots=shot_count)
    print(f"Задание IBM Quantum: job_id={runtime_job.job_id()}, backend={selected_backend.name}")
    primitive_result = runtime_job.result()
    hardware_counts = extract_counts_from_sampler_result(primitive_result[0])
    return hardware_counts, selected_backend.name


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Состояние Белла: Aer и опционально IBM Quantum.")
    parser.add_argument(
        "--ibm",
        action="store_true",
        help="Выполнить ту же схему на реальном бэкенде IBM (и сравнить с Aer при тех же shots).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        metavar="NAME",
        help="Имя бэкенда IBM (например ibm_fez). По умолчанию — least_busy.",
    )
    parser.add_argument(
        "--shots-ibm",
        type=int,
        default=1024,
        help="Число измерений на железе (и на Aer в режиме сравнения).",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    command_line_arguments = parse_command_line()

    project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / ".env")

    output_directory = ensure_output_directory()
    bell_circuit = build_bell_state_circuit()

    print(bell_circuit.draw(output="text"))

    diagram_path = save_circuit_diagram(bell_circuit, output_directory)
    print(f"Схема сохранена: {diagram_path}")

    aer_backend = AerSimulator()

    if command_line_arguments.ibm:
        if not os.environ.get("QISKIT_IBM_TOKEN", "").strip():
            print(
                "Нужен QISKIT_IBM_TOKEN (файл .env или переменная окружения). "
                "Сначала выполните init_ibm_quantum.py."
            )
            raise SystemExit(1)

        comparison_shot_count = command_line_arguments.shots_ibm
        aer_counts_for_comparison = run_shot_statistics_on_aer(
            bell_circuit,
            aer_backend,
            comparison_shot_count,
        )
        print(f"Aer, shots={comparison_shot_count}: {aer_counts_for_comparison}")

        hardware_counts, hardware_backend_label = run_on_ibm_quantum_hardware(
            bell_circuit,
            comparison_shot_count,
            command_line_arguments.backend,
        )
        print(f"IBM {hardware_backend_label}, shots={comparison_shot_count}: {hardware_counts}")

        comparison_figure = plot_histogram(
            [aer_counts_for_comparison, hardware_counts],
            legend=["Aer (идеал)", f"IBM {hardware_backend_label}"],
            figsize=(10, 5),
        )
        comparison_path = output_directory / "bell_histogram_aer_vs_ibmq.png"
        comparison_figure.savefig(comparison_path, dpi=150, bbox_inches="tight")
        comparison_figure.clear()
        print(f"Сравнение Aer / железо сохранено: {comparison_path}")
        return

    small_shot_count = 1_000
    large_shot_count = 100_000

    small_shot_counts = run_shot_statistics_on_aer(
        bell_circuit,
        aer_backend,
        small_shot_count,
    )
    large_shot_counts = run_shot_statistics_on_aer(
        bell_circuit,
        aer_backend,
        large_shot_count,
    )

    print(f"Распределение при shots={small_shot_count}: {small_shot_counts}")
    print(f"Распределение при shots={large_shot_count}: {large_shot_counts}")

    comparison_histogram = plot_histogram(
        [small_shot_counts, large_shot_counts],
        legend=["1000 измерений", "100000 измерений"],
        figsize=(10, 5),
    )
    histogram_path = output_directory / "week01_bell_histogram_comparison.png"
    comparison_histogram.savefig(histogram_path, dpi=150, bbox_inches="tight")
    comparison_histogram.clear()
    print(f"Гистограмма сравнения сохранена: {histogram_path}")


if __name__ == "__main__":
    main()
