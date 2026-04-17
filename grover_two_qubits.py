"""
Grover на 2 кубитах: поиск помеченного состояния |11⟩ (одна итерация).
Оракул: фазовый сдвиг π на |11⟩ (CZ). Диффузия — явная 2-кубитная схема (см. append_two_qubit_diffuser).
Визуализация вероятностей по шагам: grover_visualize_steps.py.

Запуск:
- только Aer (гистограмма в output/grover_histogram_aer.png):
  python grover_two_qubits.py
- Aer + IBM Quantum, сравнение гистограмм (нужен QISKIT_IBM_TOKEN в .env):
  python grover_two_qubits.py --ibm
  опционально: --backend ИМЯ --shots-ibm 1024 --optimization-level 1
"""

from __future__ import annotations

import argparse
import os
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


def append_two_qubit_diffuser(circuit: QuantumCircuit) -> None:
    circuit.h([0, 1])
    circuit.x([0, 1])
    circuit.h(1)
    circuit.cx(0, 1)
    circuit.h(1)
    circuit.x([0, 1])
    circuit.h([0, 1])


def build_grover_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h([0, 1])
    circuit.cz(0, 1)
    append_two_qubit_diffuser(circuit)
    circuit.measure([0, 1], [0, 1])
    circuit.name = "grover_2q_mark_11_one_iter"
    return circuit


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grover (2 кубита, |11⟩): Aer и опционально IBM.")
    parser.add_argument("--ibm", action="store_true")
    parser.add_argument("--backend", type=str, default=None, metavar="NAME")
    parser.add_argument("--shots-ibm", type=int, default=1024)
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Уровень оптимизации транспиляции (0–3).",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdout()
    command_line_arguments = parse_command_line()

    project_root = Path(__file__).resolve().parent
    load_dotenv_from_project(project_root)
    output_directory = ensure_output_directory(project_root)

    grover_circuit = build_grover_circuit()
    print(grover_circuit.draw(output="text"))

    diagram_path = save_circuit_figure(
        grover_circuit,
        output_directory / "grover_two_qubits_circuit.png",
    )
    print(f"Схема сохранена: {diagram_path}")

    aer_backend = AerSimulator()

    if command_line_arguments.ibm:
        if not os.environ.get("QISKIT_IBM_TOKEN", "").strip():
            print("Нужен QISKIT_IBM_TOKEN в .env.")
            raise SystemExit(1)

        shot_count = command_line_arguments.shots_ibm
        aer_counts = run_shot_statistics_on_aer(grover_circuit, aer_backend, shot_count)
        print(f"Aer, shots={shot_count}: {aer_counts}")

        hardware_counts, backend_label = run_on_ibm_quantum_hardware(
            grover_circuit,
            shot_count,
            command_line_arguments.backend,
            min_num_qubits=2,
            optimization_level=command_line_arguments.optimization_level,
        )
        print(f"IBM {backend_label}: {hardware_counts}")

        figure = plot_histogram(
            [aer_counts, hardware_counts],
            legend=["Aer (идеал)", f"IBM {backend_label}"],
            figsize=(9, 5),
        )
        out_path = output_directory / "grover_histogram_aer_vs_ibmq.png"
        figure.savefig(out_path, dpi=150, bbox_inches="tight")
        figure.clear()
        print(f"Гистограмма сохранена: {out_path}")
        return

    shot_count = 2_048
    aer_counts = run_shot_statistics_on_aer(grover_circuit, aer_backend, shot_count)
    print(f"Aer, shots={shot_count}: {aer_counts}")
    figure = plot_histogram(aer_counts, figsize=(8, 4))
    out_path = output_directory / "grover_histogram_aer.png"
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    figure.clear()
    print(f"Гистограмма сохранена: {out_path}")


if __name__ == "__main__":
    main()
