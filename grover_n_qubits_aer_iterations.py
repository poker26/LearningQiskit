"""
Гровер: одна метка |11…1⟩, идеальная симуляция (Statevector).

График P(метка) vs k и врезка: число запросов к оракулу f у Гровера (~k) vs классический
перебор в худшем случае (N запросов). По умолчанию n=12, N=4096.

Теория: θ = arcsin(1/√N), P_k = sin²((2k+1)θ); эвристика k ≈ (π/4)√N.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from qiskit import QuantumCircuit
from qiskit.circuit.library.grover_operator import GroverOperator
from qiskit.circuit.library.standard_gates import ZGate
from qiskit.quantum_info import Statevector

from ibmq_experiment import configure_utf8_stdout, ensure_output_directory


def build_phase_oracle_marked_all_ones(num_qubits: int) -> QuantumCircuit:
    """Фазовый оракул: −1 на базисе |1⟩^{⊗ n} (все кубиты в |1⟩)."""
    oracle_circuit = QuantumCircuit(num_qubits)
    oracle_circuit.append(ZGate().control(num_qubits - 1), list(range(num_qubits)))
    oracle_circuit.name = "oracle_phase_all_ones"
    return oracle_circuit


def compute_default_max_iterations(num_qubits: int) -> int:
    """Верх k с запасом после первого пика (для иллюстрации «перескока»)."""
    search_space_size = 2**num_qubits
    grover_heuristic = (np.pi / 4.0) * np.sqrt(search_space_size)
    return max(15, int(np.ceil(grover_heuristic * 1.35)))


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Гровер: график P(метка) vs k и сравнение запросов f с классикой (Statevector).",
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=12,
        metavar="N",
        help="Число кубитов n (N=2^n). По умолчанию 12.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        metavar="K",
        help="Максимум k (включительно). По умолчанию — от n.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Куда сохранить PNG (по умолчанию output/grover_{n}q_iterations_aer.png).",
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

    num_qubits = command_line_arguments.num_qubits
    if num_qubits < 2:
        raise SystemExit("--num-qubits должен быть >= 2.")

    max_iterations = command_line_arguments.max_iterations
    if max_iterations is None:
        max_iterations = compute_default_max_iterations(num_qubits)
    if max_iterations < 0:
        raise SystemExit("--max-iterations должен быть >= 0.")

    search_space_size = 2**num_qubits
    marked_state_label = "1" * num_qubits

    project_root = Path(__file__).resolve().parent
    output_directory = ensure_output_directory(project_root)
    output_path = command_line_arguments.output
    if output_path is None:
        output_path = output_directory / f"grover_{num_qubits}q_iterations_aer.png"

    theta = float(np.arcsin(1.0 / np.sqrt(search_space_size)))
    optimal_iterations_continuous = float(np.pi / (4.0 * theta))
    grover_sqrt_n_heuristic = (np.pi / 4.0) * np.sqrt(search_space_size)
    classical_worst_case_queries = int(search_space_size)

    oracle_circuit = build_phase_oracle_marked_all_ones(num_qubits)
    grover_iteration = GroverOperator(oracle_circuit, insert_barriers=False)

    preparation_circuit = QuantumCircuit(num_qubits)
    preparation_circuit.h(range(num_qubits))
    state_vector = Statevector(preparation_circuit)

    iteration_indices: list[int] = []
    probability_by_iteration: list[float] = []
    theory_probability_by_iteration: list[float] = []

    for iteration_count in range(0, max_iterations + 1):
        if iteration_count > 0:
            state_vector = state_vector.evolve(grover_iteration)
        probability_marked = float(state_vector.probabilities_dict().get(marked_state_label, 0.0))
        theory_probability = float(np.sin((2 * iteration_count + 1) * theta) ** 2)
        iteration_indices.append(iteration_count)
        probability_by_iteration.append(probability_marked)
        theory_probability_by_iteration.append(theory_probability)

    max_abs_error = float(
        np.max(np.abs(np.array(probability_by_iteration) - np.array(theory_probability_by_iteration)))
    )

    peak_index = int(np.argmax(probability_by_iteration))
    grover_oracle_queries_at_peak = int(peak_index)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(11, 5.6))
    axis.plot(
        iteration_indices,
        probability_by_iteration,
        "o-",
        color="#4C72B0",
        linewidth=2,
        markersize=4 if max_iterations > 40 else 6,
        label="Statevector (идеальная эволюция)",
    )
    axis.plot(
        iteration_indices,
        theory_probability_by_iteration,
        ":",
        color="#999999",
        linewidth=1.5,
        label=r"теория: $\sin^2((2k+1)\theta)$",
    )
    axis.axvline(
        grover_sqrt_n_heuristic,
        color="#C44E52",
        linestyle="--",
        linewidth=1.8,
        label=rf"$\frac{{\pi}}{{4}}\sqrt{{N}} \approx {grover_sqrt_n_heuristic:.2f}$",
    )
    axis.set_xlabel("Число итераций Гровера $k$ (на итерацию — один запрос к оракулу $f$)")
    axis.set_ylabel(rf"$P(|{marked_state_label}\rangle)$")
    axis.set_title(
        rf"Гровер: $n={num_qubits}$, $N={search_space_size}$, одна метка $|{marked_state_label}\rangle$",
        fontsize=11,
    )
    axis.legend(loc="lower left", fontsize=8)
    axis.set_ylim(-0.02, 1.05)
    axis.grid(True, alpha=0.3)
    tick_step = max(1, max_iterations // 12)
    axis.set_xticks(list(range(0, max_iterations + 1, tick_step)))

    inset_axis = inset_axes(
        axis,
        width="44%",
        height="42%",
        loc="upper right",
        borderpad=1.2,
    )
    bar_heights = [
        max(grover_oracle_queries_at_peak, 1),
        max(classical_worst_case_queries, 1),
    ]
    bar_colors = ["#4C72B0", "#C44E52"]
    row_positions = np.arange(2)
    inset_axis.barh(row_positions, bar_heights, color=bar_colors, height=0.55, align="center")
    inset_axis.set_xscale("log")
    inset_axis.set_yticks(row_positions)
    inset_axis.set_yticklabels(
        [
            rf"Гровер: $k={grover_oracle_queries_at_peak}$",
            rf"Классика: $N={classical_worst_case_queries}$",
        ],
        fontsize=8,
    )
    inset_axis.set_xlabel("Запросов к $f$ (лог. шкала)", fontsize=8)
    inset_axis.set_title("Худший случай классики vs пик Гровера", fontsize=9, fontweight="bold")
    inset_axis.grid(True, axis="x", alpha=0.3)
    for row_index, bar_value in enumerate(bar_heights):
        inset_axis.text(
            bar_value * 1.08,
            row_index,
            f"{bar_value}",
            va="center",
            fontsize=8,
            color="#333333",
        )

    figure.text(
        0.5,
        0.03,
        (
            f"max |симуляция − теория| = {max_abs_error:.2e}. "
            f"π/(4θ) ≈ {optimal_iterations_continuous:.2f}. "
            "Классика: модель «каждый кандидат — отдельный вызов $f$», худший случай до $N$ вызовов."
        ),
        ha="center",
        fontsize=8,
        style="italic",
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.14)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Сохранено: {output_path}")
    print(
        f"Пик (симуляция): k={peak_index}, P={probability_by_iteration[peak_index]:.6f}",
    )
    print(
        f"Эвристика (π/4)·√N = {grover_sqrt_n_heuristic:.4f}; "
        f"классика (худший случай): {classical_worst_case_queries} запросов f",
    )


if __name__ == "__main__":
    main()
