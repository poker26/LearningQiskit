"""
Гровер на 6 кубитах (N=64): одна фиксированная метка |111111⟩.
Симуляция идеальной эволюции (Statevector / Aer без шума) и график P(метка) vs k.

Теория для одной метки: θ = arcsin(1/√N), P_k = sin²((2k+1)θ); ориентир k ≈ (π/4)√N.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library.grover_operator import GroverOperator
from qiskit.circuit.library.standard_gates import ZGate
from qiskit.quantum_info import Statevector

from ibmq_experiment import configure_utf8_stdout, ensure_output_directory

NUM_QUBITS = 6
MARKED_STATE_LABEL = "111111"
SEARCH_SPACE_SIZE = 2**NUM_QUBITS


def build_phase_oracle_marked_all_ones() -> QuantumCircuit:
    """Фазовый оракул: −1 на базисе |11…1⟩ (все шесть кубитов в |1⟩)."""
    oracle_circuit = QuantumCircuit(NUM_QUBITS)
    oracle_circuit.append(ZGate().control(NUM_QUBITS - 1), list(range(NUM_QUBITS)))
    oracle_circuit.name = "oracle_phase_all_ones"
    return oracle_circuit


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Гровер 6q: график вероятности метки vs число итераций (Aer/Statevector).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        metavar="K",
        help="Максимум k (включительно): 0…K.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Куда сохранить PNG (по умолчанию output/grover_6q_iterations_aer.png).",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*GroverOperator.*",
        category=DeprecationWarning,
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    configure_utf8_stdout()
    command_line_arguments = parse_command_line()

    if command_line_arguments.max_iterations < 0:
        raise SystemExit("--max-iterations должен быть >= 0.")

    project_root = Path(__file__).resolve().parent
    output_directory = ensure_output_directory(project_root)
    output_path = command_line_arguments.output
    if output_path is None:
        output_path = output_directory / "grover_6q_iterations_aer.png"

    theta = float(np.arcsin(1.0 / np.sqrt(SEARCH_SPACE_SIZE)))
    optimal_iterations_continuous = float(np.pi / (4.0 * theta))
    grover_sqrt_n_heuristic = (np.pi / 4.0) * np.sqrt(SEARCH_SPACE_SIZE)

    oracle_circuit = build_phase_oracle_marked_all_ones()
    grover_iteration = GroverOperator(oracle_circuit, insert_barriers=False)

    preparation_circuit = QuantumCircuit(NUM_QUBITS)
    preparation_circuit.h(range(NUM_QUBITS))
    state_vector = Statevector(preparation_circuit)

    iteration_indices: list[int] = []
    probability_by_iteration: list[float] = []
    theory_probability_by_iteration: list[float] = []

    for iteration_count in range(0, command_line_arguments.max_iterations + 1):
        if iteration_count > 0:
            state_vector = state_vector.evolve(grover_iteration)
        probability_marked = float(state_vector.probabilities_dict().get(MARKED_STATE_LABEL, 0.0))
        theory_probability = float(np.sin((2 * iteration_count + 1) * theta) ** 2)
        iteration_indices.append(iteration_count)
        probability_by_iteration.append(probability_marked)
        theory_probability_by_iteration.append(theory_probability)

    max_abs_error = float(
        np.max(np.abs(np.array(probability_by_iteration) - np.array(theory_probability_by_iteration)))
    )

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(10, 5.2))
    axis.plot(
        iteration_indices,
        probability_by_iteration,
        "o-",
        color="#4C72B0",
        linewidth=2,
        markersize=6,
        label="Statevector (идеальная эволюция)",
    )
    axis.plot(
        iteration_indices,
        theory_probability_by_iteration,
        ":",
        color="#999999",
        linewidth=1.5,
        label=r"теория: $\sin^2((2k+1)\theta)$, $\sin\theta=1/\sqrt{N}$",
    )
    axis.axvline(
        grover_sqrt_n_heuristic,
        color="#C44E52",
        linestyle="--",
        linewidth=1.8,
        label=rf"$\frac{{\pi}}{{4}}\sqrt{{N}} \approx {grover_sqrt_n_heuristic:.2f}$",
    )
    axis.set_xlabel("Число итераций Гровера $k$")
    axis.set_ylabel(rf"$P(|{MARKED_STATE_LABEL}\rangle)$")
    axis.set_title(
        rf"Гровер: $n={NUM_QUBITS}$, $N={SEARCH_SPACE_SIZE}$, метка $|{MARKED_STATE_LABEL}\rangle$ "
        r"(оракул — многократно контролируемый $Z$)",
        fontsize=11,
    )
    axis.legend(loc="upper right", fontsize=8)
    axis.set_xticks(iteration_indices)
    axis.set_ylim(-0.02, 1.05)
    axis.grid(True, alpha=0.3)
    figure.text(
        0.5,
        0.01,
        (
            f"max |симуляция − теория| = {max_abs_error:.2e}. "
            f"Непрерывный оптимум по θ: π/(4θ) ≈ {optimal_iterations_continuous:.2f} "
            "(для конечного N чуть отличается от (π/4)√N). "
            "После пика P падает — «перескок» итераций."
        ),
        ha="center",
        fontsize=8,
        style="italic",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    peak_index = int(np.argmax(probability_by_iteration))
    print(f"Сохранено: {output_path}")
    print(
        f"Пик (симуляция): k={peak_index}, P={probability_by_iteration[peak_index]:.6f}",
    )
    print(f"Эвристика (π/4)·√N = {grover_sqrt_n_heuristic:.4f}; непрерывный оптимум π/(4θ) = {optimal_iterations_continuous:.4f}")


if __name__ == "__main__":
    main()
