"""
VQE для молекулы водорода H₂: базис STO-3G, равновесная геометрия (~0.735 Å),
отображение Jordan–Wigner, 4 кубита.

На Windows PySCF часто не ставится без компилятора — по умолчанию используется
встроенный электронный гамильтониан из h2_sto3g_jordan_wigner_hamiltonian.py
(совпадает с выводом Qiskit Nature + PySCF).

На Linux можно пересчитать интегралы: python vqe_h2_sto3g.py --use-pyscf

Минимизируется электронная энергия ⟨H_el⟩ (Hartree); для полной энергии прибавляется
ядро E_nuc (печатается и линия на графике при желании).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from h2_sto3g_jordan_wigner_hamiltonian import (
    H2_STO3G_EQUILIBRIUM_NUCLEAR_REPULSION_HARTREE,
    build_h2_sto3g_jordan_wigner_hamiltonian_builtin,
)
from ibmq_experiment import configure_utf8_stdout, ensure_output_directory


def configure_matplotlib_cyrillic() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def try_build_hamiltonian_from_pyscf() -> tuple[SparsePauliOp, float]:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit

    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    nuclear_repulsion = float(problem.nuclear_repulsion_energy)
    return qubit_op, nuclear_repulsion


def build_variational_ansatz_h2() -> tuple:
    """Hartree–Fock + RealAmplitudes (как простой химический анзац)."""
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    mapper = JordanWignerMapper()
    hartree_fock_circuit = HartreeFock(2, (1, 1), mapper)
    layered_ansatz = RealAmplitudes(
        num_qubits=4,
        entanglement="linear",
        reps=3,
        insert_barriers=False,
    )
    combined_ansatz = hartree_fock_circuit.compose(layered_ansatz)
    return combined_ansatz


def exact_min_electronic_energy_hartree(qubit_hamiltonian: SparsePauliOp) -> float:
    hermitian_matrix = qubit_hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvalsh(hermitian_matrix)
    return float(np.min(eigenvalues.real))


def save_convergence_figure(
    expectation_history: list[float],
    exact_electronic_energy: float,
    nuclear_repulsion_energy: float,
    output_path: Path,
) -> None:
    configure_matplotlib_cyrillic()
    evaluation_index_array = np.arange(1, len(expectation_history) + 1)
    running_minimum = np.minimum.accumulate(np.array(expectation_history, dtype=np.float64))
    total_exact = exact_electronic_energy + nuclear_repulsion_energy

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(
        evaluation_index_array,
        expectation_history,
        "o-",
        color="#7F7F7F",
        markersize=3,
        linewidth=1,
        alpha=0.85,
        label=r"$\langle H_\mathrm{el}\rangle$ каждый вызов",
    )
    axis.plot(
        evaluation_index_array,
        running_minimum,
        "-",
        color="#4C72B0",
        linewidth=2.0,
        label="лучший минимум на шаг",
    )
    axis.axhline(
        exact_electronic_energy,
        color="#C44E52",
        linestyle="--",
        linewidth=1.8,
        label=rf"точн. $E_\mathrm{{el}}$ (диаг.): {exact_electronic_energy:.6f} Ha",
    )
    axis.axhline(
        total_exact,
        color="#55A868",
        linestyle=":",
        linewidth=1.6,
        label=rf"$E_\mathrm{{el}}+E_\mathrm{{nuc}}$: {total_exact:.6f} Ha",
    )
    axis.set_xlabel(r"Номер вызова целевой функции")
    axis.set_ylabel(r"Энергия (Hartree)")
    axis.set_title(
        r"VQE: H$_2$, STO-3G, Jordan–Wigner (4 q), StatevectorEstimator + COBYLA",
        fontsize=11,
        fontweight="bold",
    )
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.14)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VQE для H₂ STO-3G (4 кубита, JW).")
    parser.add_argument(
        "--use-pyscf",
        action="store_true",
        help="Считать интегралы через PySCF (нужен pyscf; удобно на Linux).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="COBYLA maxiter.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    configure_utf8_stdout()
    command_line_arguments = parse_command_line()
    output_directory = ensure_output_directory(Path(__file__).resolve().parent)

    nuclear_repulsion_energy = H2_STO3G_EQUILIBRIUM_NUCLEAR_REPULSION_HARTREE
    hamiltonian_source_label = "встроенные коэффициенты (как Nature+PySCF)"

    if command_line_arguments.use_pyscf:
        try:
            qubit_hamiltonian, nuclear_repulsion_energy = try_build_hamiltonian_from_pyscf()
            hamiltonian_source_label = "PySCF (живой расчёт интегралов)"
        except Exception as exc:
            print(f"PySCF недоступен ({exc!r}). Используем встроенный гамильтониан.")
            qubit_hamiltonian = build_h2_sto3g_jordan_wigner_hamiltonian_builtin()
    else:
        qubit_hamiltonian = build_h2_sto3g_jordan_wigner_hamiltonian_builtin()

    exact_electronic_energy = exact_min_electronic_energy_hartree(qubit_hamiltonian)
    print(f"Источник гамильтониана: {hamiltonian_source_label}")
    print(f"Точная электронная энергия (диагонализация): {exact_electronic_energy:.8f} Ha")
    print(f"Ядерное отталкивание E_nuc: {nuclear_repulsion_energy:.8f} Ha")
    print(
        f"Точная полная энергия E_el + E_nuc: {exact_electronic_energy + nuclear_repulsion_energy:.8f} Ha",
    )

    ansatz_circuit = build_variational_ansatz_h2()
    estimator = StatevectorEstimator()
    expectation_history: list[float] = []

    def objective_expectation_value(parameter_vector: list[float]) -> float:
        bound_circuit = ansatz_circuit.assign_parameters(parameter_vector)
        job_result = estimator.run([(bound_circuit, qubit_hamiltonian)]).result()
        energy = float(job_result[0].data.evs)
        expectation_history.append(energy)
        return energy

    initial_parameters = [0.0] * ansatz_circuit.num_parameters
    optimization_result = minimize(
        objective_expectation_value,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": command_line_arguments.maxiter},
    )

    best_electronic = float(optimization_result.fun)
    print(f"VQE минимум ⟨H_el⟩: {best_electronic:.8f} Ha")
    print(f"Ошибка от точной E_el: {best_electronic - exact_electronic_energy:.2e} Ha")
    print(f"Успех COBYLA: {optimization_result.success}, nfev={optimization_result.nfev}")
    print(f"Сообщение оптимизатора: {optimization_result.message}")

    convergence_path = output_directory / "vqe_h2_sto3g_convergence.png"
    save_convergence_figure(
        expectation_history,
        exact_electronic_energy,
        nuclear_repulsion_energy,
        convergence_path,
    )
    print(f"График: {convergence_path}")


if __name__ == "__main__":
    main()
