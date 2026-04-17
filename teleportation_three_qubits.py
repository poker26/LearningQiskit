"""
Квантовая телепортация |+⟩ с q0 на q2 (классическая связь через измерения Алисы и if_test).
На Aer ожидается ~равные доли 0/1 на измерении q2 (состояние |+⟩).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
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


def build_teleportation_circuit() -> QuantumCircuit:
    quantum_register = QuantumRegister(3, "q")
    classical_alice_z = ClassicalRegister(1, "m_alice_z")
    classical_alice_x = ClassicalRegister(1, "m_alice_x")
    classical_bob = ClassicalRegister(1, "bob")
    circuit = QuantumCircuit(
        quantum_register,
        classical_alice_z,
        classical_alice_x,
        classical_bob,
    )

    circuit.h(0)
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0, classical_alice_z)
    circuit.measure(1, classical_alice_x)
    with circuit.if_test((classical_alice_z, 1)):
        circuit.z(2)
    with circuit.if_test((classical_alice_x, 1)):
        circuit.x(2)
    circuit.measure(2, classical_bob)
    circuit.name = "teleport_plus_to_q2"
    return circuit


def marginal_bob_outcome_counts(joint_counts: dict[str, int]) -> dict[str, int]:
    bob_totals: defaultdict[str, int] = defaultdict(int)
    for outcome_label, occurrence_count in joint_counts.items():
        parts = outcome_label.split()
        if len(parts) < 2:
            raise ValueError(f"Неожиданный формат исхода: {outcome_label!r}")
        bob_bit_label = parts[-1]
        bob_totals[bob_bit_label] += occurrence_count
    return dict(bob_totals)


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Телепортация (3 кубита): Aer и опционально IBM.")
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

    teleport_circuit = build_teleportation_circuit()
    print(teleport_circuit.draw(output="text"))

    diagram_path = save_circuit_figure(
        teleport_circuit,
        output_directory / "teleportation_circuit.png",
    )
    print(f"Схема сохранена: {diagram_path}")

    aer_backend = AerSimulator()

    if command_line_arguments.ibm:
        if not os.environ.get("QISKIT_IBM_TOKEN", "").strip():
            print("Нужен QISKIT_IBM_TOKEN в .env.")
            raise SystemExit(1)

        shot_count = command_line_arguments.shots_ibm
        aer_joint = run_shot_statistics_on_aer(teleport_circuit, aer_backend, shot_count)
        aer_bob = marginal_bob_outcome_counts(aer_joint)
        print(f"Aer (только bob), shots={shot_count}: {aer_bob}")

        hardware_joint, backend_label = run_on_ibm_quantum_hardware(
            teleport_circuit,
            shot_count,
            command_line_arguments.backend,
            min_num_qubits=3,
            optimization_level=command_line_arguments.optimization_level,
        )
        hardware_bob = marginal_bob_outcome_counts(hardware_joint)
        print(f"IBM {backend_label} (только bob): {hardware_bob}")

        figure = plot_histogram(
            [aer_bob, hardware_bob],
            legend=["Aer (идеал)", f"IBM {backend_label}"],
            figsize=(7, 4),
        )
        out_path = output_directory / "teleport_bob_marginal_aer_vs_ibmq.png"
        figure.savefig(out_path, dpi=150, bbox_inches="tight")
        figure.clear()
        print(f"Гистограмма (маргиналь по q2) сохранена: {out_path}")
        return

    shot_count = 4_096
    aer_joint = run_shot_statistics_on_aer(teleport_circuit, aer_backend, shot_count)
    aer_bob = marginal_bob_outcome_counts(aer_joint)
    print(f"Aer (только bob), shots={shot_count}: {aer_bob}")
    figure = plot_histogram(aer_bob, figsize=(6, 4))
    out_path = output_directory / "teleport_bob_marginal_aer.png"
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    figure.clear()
    print(f"Гистограмма сохранена: {out_path}")


if __name__ == "__main__":
    main()
