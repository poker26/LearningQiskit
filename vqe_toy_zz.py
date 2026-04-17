"""
Мини-VQE на симуляторе: гамильтониан H = Z₀Z₁, анзац RealAmplitudes, StatevectorEstimator + COBYLA.
Ожидаемая минимальная энергия (точное основное состояние): −1.
Без IBM — только локально, чтобы быстро крутить цикл оптимизации.
"""

from __future__ import annotations

from pathlib import Path

from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from ibmq_experiment import configure_utf8_stdout, ensure_output_directory, save_circuit_figure


def build_hamiltonian_zz() -> SparsePauliOp:
    return SparsePauliOp.from_list([("ZZ", 1.0)])


def main() -> None:
    configure_utf8_stdout()

    output_directory = ensure_output_directory(Path(__file__).resolve().parent)

    hamiltonian = build_hamiltonian_zz()
    ansatz_circuit = real_amplitudes(num_qubits=2, entanglement="linear", reps=1, insert_barriers=False)
    estimator = StatevectorEstimator()

    def objective_expectation_value(parameter_vector: list[float]) -> float:
        bound_circuit = ansatz_circuit.assign_parameters(parameter_vector)
        primitive_result = estimator.run([(bound_circuit, hamiltonian)]).result()
        expectation_value = float(primitive_result[0].data.evs)
        return expectation_value

    initial_parameters = [0.1] * ansatz_circuit.num_parameters
    optimization_result = minimize(
        objective_expectation_value,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": 120},
    )

    print(f"Минимальная найденная ⟨H⟩: {optimization_result.fun:.6f}")
    print(f"Ожидание (ground ZZ): −1.0")
    print(f"Успех оптимизации: {optimization_result.success}, итераций: {optimization_result.nfev}")

    best_circuit = ansatz_circuit.assign_parameters(optimization_result.x)
    best_circuit.measure_all()
    diagram_path = save_circuit_figure(
        best_circuit,
        output_directory / "vqe_toy_zz_best_ansatz_measured.png",
    )
    print(f"Схема лучшего анзаца (с измерениями) сохранена: {diagram_path}")


if __name__ == "__main__":
    main()
