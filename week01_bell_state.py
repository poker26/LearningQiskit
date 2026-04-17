"""
Неделя 1: состояние Белла на симуляторе Aer.
Сохраняет схему и гистограммы в папку output/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qiskit import QuantumCircuit
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


def run_shot_statistics(
    bell_circuit: QuantumCircuit,
    aer_backend: AerSimulator,
    shot_count: int,
) -> dict[str, int]:
    simulation_job = aer_backend.run(bell_circuit, shots=shot_count)
    measurement_result = simulation_job.result()
    outcome_counts = measurement_result.get_counts()
    return outcome_counts


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / ".env")

    output_directory = ensure_output_directory()
    bell_circuit = build_bell_state_circuit()

    print(bell_circuit.draw(output="text"))

    diagram_path = save_circuit_diagram(bell_circuit, output_directory)
    print(f"Схема сохранена: {diagram_path}")

    aer_backend = AerSimulator()

    small_shot_count = 1_000
    large_shot_count = 100_000

    small_shot_counts = run_shot_statistics(bell_circuit, aer_backend, small_shot_count)
    large_shot_counts = run_shot_statistics(bell_circuit, aer_backend, large_shot_count)

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

    if os.environ.get("QISKIT_IBM_TOKEN"):
        print(
            "Обнаружен QISKIT_IBM_TOKEN. На неделе 2 подключим реальный бэкенд; "
            "сейчас скрипт намеренно использует только локальный Aer."
        )


if __name__ == "__main__":
    main()
