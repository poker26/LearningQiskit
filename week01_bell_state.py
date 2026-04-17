"""
Состояние Белла: симулятор Aer и (опционально) IBM Quantum через SamplerV2.
Сохраняет схему и гистограммы в папку output/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

from ibmq_experiment import (
    configure_utf8_stdout,
    ensure_output_directory,
    load_dotenv_from_project,
    run_on_ibm_quantum_hardware,
    run_shot_statistics_on_aer,
    save_circuit_figure,
)


def build_bell_state_circuit() -> QuantumCircuit:
    bell_circuit = QuantumCircuit(2, 2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_circuit.measure([0, 1], [0, 1])
    bell_circuit.name = "bell_phi_plus"
    return bell_circuit


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
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Уровень оптимизации транспиляции под выбранный чип (0–3).",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdout()

    command_line_arguments = parse_command_line()

    project_root = Path(__file__).resolve().parent
    load_dotenv_from_project(project_root)

    output_directory = ensure_output_directory(project_root)
    bell_circuit = build_bell_state_circuit()

    print(bell_circuit.draw(output="text"))

    diagram_path = save_circuit_figure(
        bell_circuit,
        output_directory / "week01_bell_circuit.png",
    )
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
            min_num_qubits=2,
            optimization_level=command_line_arguments.optimization_level,
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
