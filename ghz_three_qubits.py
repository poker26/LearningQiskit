"""
GHZ на трёх кубитах: (|000⟩ + |111⟩) / √2 на Aer и опционально на IBM Quantum.
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


def build_ghz_circuit() -> QuantumCircuit:
    ghz_circuit = QuantumCircuit(3, 3)
    ghz_circuit.h(0)
    ghz_circuit.cx(0, 1)
    ghz_circuit.cx(1, 2)
    ghz_circuit.measure([0, 1, 2], [0, 1, 2])
    ghz_circuit.name = "ghz_3q"
    return ghz_circuit


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GHZ(3): Aer и опционально IBM Quantum.")
    parser.add_argument("--ibm", action="store_true", help="Сравнить Aer и реальный бэкенд IBM.")
    parser.add_argument("--backend", type=str, default=None, metavar="NAME", help="Имя бэкенда IBM.")
    parser.add_argument(
        "--shots-ibm",
        type=int,
        default=1024,
        help="Shots для IBM и для Aer в режиме сравнения.",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdout()
    command_line_arguments = parse_command_line()

    project_root = Path(__file__).resolve().parent
    load_dotenv_from_project(project_root)
    output_directory = ensure_output_directory(project_root)

    ghz_circuit = build_ghz_circuit()
    print(ghz_circuit.draw(output="text"))

    diagram_path = save_circuit_figure(
        ghz_circuit,
        output_directory / "ghz_three_qubits_circuit.png",
    )
    print(f"Схема сохранена: {diagram_path}")

    aer_backend = AerSimulator()

    if command_line_arguments.ibm:
        if not os.environ.get("QISKIT_IBM_TOKEN", "").strip():
            print("Нужен QISKIT_IBM_TOKEN в .env (см. init_ibm_quantum.py).")
            raise SystemExit(1)

        shot_count = command_line_arguments.shots_ibm
        aer_counts = run_shot_statistics_on_aer(ghz_circuit, aer_backend, shot_count)
        print(f"Aer, shots={shot_count}: {aer_counts}")

        hardware_counts, backend_label = run_on_ibm_quantum_hardware(
            ghz_circuit,
            shot_count,
            command_line_arguments.backend,
            min_num_qubits=3,
        )
        print(f"IBM {backend_label}, shots={shot_count}: {hardware_counts}")

        figure = plot_histogram(
            [aer_counts, hardware_counts],
            legend=["Aer (идеал)", f"IBM {backend_label}"],
            figsize=(11, 5),
        )
        out_path = output_directory / "ghz_histogram_aer_vs_ibmq.png"
        figure.savefig(out_path, dpi=150, bbox_inches="tight")
        figure.clear()
        print(f"Гистограмма сохранена: {out_path}")
        return

    small_shots = 1_000
    large_shots = 100_000
    counts_small = run_shot_statistics_on_aer(ghz_circuit, aer_backend, small_shots)
    counts_large = run_shot_statistics_on_aer(ghz_circuit, aer_backend, large_shots)
    print(f"Aer shots={small_shots}: {counts_small}")
    print(f"Aer shots={large_shots}: {counts_large}")

    figure = plot_histogram(
        [counts_small, counts_large],
        legend=["1000 измерений", "100000 измерений"],
        figsize=(11, 5),
    )
    out_path = output_directory / "ghz_histogram_aer_shots.png"
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    figure.clear()
    print(f"Гистограмма сохранена: {out_path}")


if __name__ == "__main__":
    main()
