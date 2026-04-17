"""
Мини-VQE на симуляторе: гамильтониан H = Z₀Z₁, анзац RealAmplitudes, StatevectorEstimator + COBYLA.
Ожидаемая минимальная энергия (точное основное состояние): −1.
Без IBM — только локально, чтобы быстро крутить цикл оптимизации.
Сохраняет PNG со сходимостью ⟨H⟩ по вызовам целевой функции.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from ibmq_experiment import configure_utf8_stdout, ensure_output_directory, save_circuit_figure


def build_hamiltonian_zz() -> SparsePauliOp:
    return SparsePauliOp.from_list([("ZZ", 1.0)])


def configure_matplotlib_cyrillic() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def save_convergence_figure(
    expectation_by_evaluation: list[float],
    exact_ground_energy: float,
    output_path: Path,
) -> None:
    configure_matplotlib_cyrillic()
    evaluation_index_array = np.arange(1, len(expectation_by_evaluation) + 1)
    running_minimum_by_evaluation = np.minimum.accumulate(np.array(expectation_by_evaluation, dtype=np.float64))

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        evaluation_index_array,
        expectation_by_evaluation,
        "o-",
        color="#7F7F7F",
        markersize=4,
        linewidth=1,
        alpha=0.85,
        label=r"$\langle H\rangle$ при каждом вызове",
    )
    axis.plot(
        evaluation_index_array,
        running_minimum_by_evaluation,
        "-",
        color="#4C72B0",
        linewidth=2.2,
        label="минимум на текущий шаг (лучшее из увиденного)",
    )
    axis.axhline(
        exact_ground_energy,
        color="#C44E52",
        linestyle="--",
        linewidth=1.8,
        label=rf"точный минимум $ZZ$: {exact_ground_energy:g}",
    )
    axis.set_xlabel(r"Номер вызова целевой функции (оценка $\langle H\rangle$)")
    axis.set_ylabel(r"$\langle H\rangle = \langle Z_0 Z_1\rangle$")
    axis.set_title(
        "VQE (игрушка): сходимость на StatevectorEstimator, оптимизатор COBYLA",
        fontsize=12,
        fontweight="bold",
    )
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(True, alpha=0.3)
    axis.margins(x=0.02)
    figure.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.14)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_utf8_stdout()

    output_directory = ensure_output_directory(Path(__file__).resolve().parent)

    hamiltonian = build_hamiltonian_zz()
    ansatz_circuit = real_amplitudes(num_qubits=2, entanglement="linear", reps=1, insert_barriers=False)
    estimator = StatevectorEstimator()

    expectation_history: list[float] = []

    def objective_expectation_value(parameter_vector: list[float]) -> float:
        bound_circuit = ansatz_circuit.assign_parameters(parameter_vector)
        primitive_result = estimator.run([(bound_circuit, hamiltonian)]).result()
        expectation_value = float(primitive_result[0].data.evs)
        expectation_history.append(expectation_value)
        return expectation_value

    initial_parameters = [0.1] * ansatz_circuit.num_parameters
    optimization_result = minimize(
        objective_expectation_value,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": 120},
    )

    exact_ground_energy = -1.0
    print(f"Минимальная найденная ⟨H⟩: {optimization_result.fun:.6f}")
    print(f"Ожидание (ground ZZ): {exact_ground_energy}")
    print(f"Успех оптимизации: {optimization_result.success}, итераций: {optimization_result.nfev}")

    convergence_plot_path = output_directory / "vqe_toy_zz_convergence.png"
    save_convergence_figure(expectation_history, exact_ground_energy, convergence_plot_path)
    print(f"График сходимости: {convergence_plot_path}")

    best_circuit = ansatz_circuit.assign_parameters(optimization_result.x)
    best_circuit.measure_all()
    diagram_path = save_circuit_figure(
        best_circuit,
        output_directory / "vqe_toy_zz_best_ansatz_measured.png",
    )
    print(f"Схема лучшего анзаца (с измерениями) сохранена: {diagram_path}")


if __name__ == "__main__":
    main()
