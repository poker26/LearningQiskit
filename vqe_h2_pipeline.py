"""
Конвейер H₂ VQE (STO-3G, JW, 4 кубита): идеальный Statevector → шумный Aer
(NoiseModel от GenericBackendV2) → реальное железо IBM Quantum (EstimatorV2 + transpile).

Полный прогон по умолчанию: идеал → шум → одна оценка энергии на железе в лучшей точке θ
(без полного VQE на устройстве — это десятки/сотни дорогих заданий).

Примеры:
  python vqe_h2_pipeline.py
  python vqe_h2_pipeline.py --stages ideal noisy --maxiter-ideal 120 --maxiter-noisy 40
  python vqe_h2_pipeline.py --stages ibm --params-json output/vqe_h2_pipeline_best_params.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from scipy.optimize import minimize

from h2_sto3g_jordan_wigner_hamiltonian import (
    H2_STO3G_EQUILIBRIUM_NUCLEAR_REPULSION_HARTREE,
    build_h2_sto3g_jordan_wigner_hamiltonian_builtin,
)
from ibmq_experiment import (
    configure_utf8_stdout,
    ensure_output_directory,
    load_dotenv_from_project,
)
from vqe_h2_sto3g import (
    build_variational_ansatz_h2,
    exact_min_electronic_energy_hartree,
    save_convergence_figure,
    try_build_hamiltonian_from_pyscf,
)


def read_primitive_expectation_value(primitive_result, pub_index: int = 0) -> float:
    expectation_values = primitive_result[pub_index].data.evs
    return float(np.asarray(expectation_values).reshape(-1)[0])


def load_qubit_hamiltonian(use_pyscf: bool) -> tuple[SparsePauliOp, float, str]:
    nuclear_repulsion_energy = H2_STO3G_EQUILIBRIUM_NUCLEAR_REPULSION_HARTREE
    hamiltonian_source_label = "встроенные коэффициенты (как Nature+PySCF)"
    if use_pyscf:
        try:
            qubit_hamiltonian, nuclear_repulsion_energy = try_build_hamiltonian_from_pyscf()
            hamiltonian_source_label = "PySCF (живой расчёт интегралов)"
        except Exception as exc:
            print(f"PySCF недоступен ({exc!r}). Используем встроенный гамильтониан.")
            qubit_hamiltonian = build_h2_sto3g_jordan_wigner_hamiltonian_builtin()
    else:
        qubit_hamiltonian = build_h2_sto3g_jordan_wigner_hamiltonian_builtin()
    return qubit_hamiltonian, nuclear_repulsion_energy, hamiltonian_source_label


def run_stage_ideal_statevector(
    ansatz_circuit,
    qubit_hamiltonian: SparsePauliOp,
    nuclear_repulsion_energy: float,
    exact_electronic_energy: float,
    maxiter: int,
    output_directory: Path,
) -> dict:
    estimator = StatevectorEstimator()
    expectation_history: list[float] = []

    def objective_expectation_value(parameter_vector: list[float]) -> float:
        bound_circuit = ansatz_circuit.assign_parameters(parameter_vector)
        job_result = estimator.run([(bound_circuit, qubit_hamiltonian)]).result()
        energy = read_primitive_expectation_value(job_result, 0)
        expectation_history.append(energy)
        return energy

    initial_parameters = [0.0] * ansatz_circuit.num_parameters
    optimization_result = minimize(
        objective_expectation_value,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": maxiter},
    )
    best_parameters = list(np.asarray(optimization_result.x, dtype=float))
    best_electronic = float(optimization_result.fun)
    convergence_path = output_directory / "vqe_h2_pipeline_ideal_convergence.png"
    save_convergence_figure(
        expectation_history,
        exact_electronic_energy,
        nuclear_repulsion_energy,
        convergence_path,
    )
    return {
        "stage": "ideal",
        "best_electronic_energy_hartree": best_electronic,
        "best_parameters": best_parameters,
        "optimizer_success": bool(optimization_result.success),
        "optimizer_nfev": int(optimization_result.nfev),
        "convergence_png": str(convergence_path),
        "expectation_history_length": len(expectation_history),
    }


def build_noisy_simulation_estimator(
    noisy_shots: int,
    random_seed: int | None,
) -> tuple[AerEstimatorV2, GenericBackendV2, object]:
    fake_backend_for_noise = GenericBackendV2(num_qubits=4)
    noise_model = NoiseModel.from_backend(fake_backend_for_noise)
    backend_options: dict = {"noise_model": noise_model}
    run_options: dict = {"shots": noisy_shots}
    if random_seed is not None:
        run_options["seed_simulator"] = random_seed
    noisy_estimator = AerEstimatorV2(
        options={
            "backend_options": backend_options,
            "run_options": run_options,
        }
    )
    return noisy_estimator, fake_backend_for_noise, noise_model


def run_stage_noisy_aer(
    ansatz_circuit,
    qubit_hamiltonian: SparsePauliOp,
    nuclear_repulsion_energy: float,
    exact_electronic_energy: float,
    maxiter: int,
    noisy_shots: int,
    random_seed: int | None,
    initial_parameter_vector: list[float] | None,
    output_directory: Path,
) -> dict:
    noisy_estimator, fake_backend_for_noise, _noise_model = build_noisy_simulation_estimator(
        noisy_shots=noisy_shots,
        random_seed=random_seed,
    )
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        target=fake_backend_for_noise.target,
    )
    expectation_history: list[float] = []

    def objective_noisy_expectation(parameter_vector: list[float]) -> float:
        bound_circuit = ansatz_circuit.assign_parameters(parameter_vector)
        transpiled_circuit = pass_manager.run(bound_circuit)
        job_result = noisy_estimator.run([(transpiled_circuit, qubit_hamiltonian)]).result()
        energy = read_primitive_expectation_value(job_result, 0)
        expectation_history.append(energy)
        return energy

    if initial_parameter_vector is None:
        initial_parameters = [0.0] * ansatz_circuit.num_parameters
    else:
        initial_parameters = list(initial_parameter_vector)

    optimization_result = minimize(
        objective_noisy_expectation,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": maxiter},
    )
    best_parameters = list(np.asarray(optimization_result.x, dtype=float))
    best_electronic = float(optimization_result.fun)
    convergence_path = output_directory / "vqe_h2_pipeline_noisy_convergence.png"
    save_convergence_figure(
        expectation_history,
        exact_electronic_energy,
        nuclear_repulsion_energy,
        convergence_path,
    )
    return {
        "stage": "noisy",
        "best_electronic_energy_hartree": best_electronic,
        "best_parameters": best_parameters,
        "optimizer_success": bool(optimization_result.success),
        "optimizer_nfev": int(optimization_result.nfev),
        "convergence_png": str(convergence_path),
        "expectation_history_length": len(expectation_history),
        "noisy_shots": noisy_shots,
    }


def select_ibm_quantum_backend(runtime_service, backend_name: str | None, min_num_qubits: int):
    if backend_name:
        return runtime_service.backend(backend_name)
    return runtime_service.least_busy(
        min_num_qubits=min_num_qubits,
        operational=True,
        simulator=False,
    )


def run_stage_ibm_quantum(
    ansatz_circuit,
    qubit_hamiltonian: SparsePauliOp,
    parameter_vector: list[float],
    backend_name: str | None,
    optimization_level: int,
    ibm_maxiter: int,
) -> dict:
    from qiskit_ibm_runtime import EstimatorV2 as IbmEstimatorV2, QiskitRuntimeService

    load_dotenv_from_project(Path(__file__).resolve().parent)
    quantum_channel = "ibm_quantum_platform"
    runtime_service = QiskitRuntimeService(channel=quantum_channel)
    selected_backend = select_ibm_quantum_backend(
        runtime_service,
        backend_name,
        min_num_qubits=max(4, ansatz_circuit.num_qubits),
    )
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        target=selected_backend.target,
    )
    hardware_estimator = IbmEstimatorV2(mode=selected_backend)

    def evaluate_energy_at_parameters(theta: list[float]) -> float:
        bound_circuit = ansatz_circuit.assign_parameters(theta)
        transpiled_circuit = pass_manager.run(bound_circuit)
        mapped_hamiltonian = qubit_hamiltonian.apply_layout(
            transpiled_circuit.layout,
            num_qubits=transpiled_circuit.num_qubits,
        )
        runtime_job = hardware_estimator.run([(transpiled_circuit, mapped_hamiltonian)])
        print(
            f"Задание IBM Quantum Estimator: job_id={runtime_job.job_id()}, "
            f"backend={selected_backend.name}",
        )
        primitive_result = runtime_job.result()
        return read_primitive_expectation_value(primitive_result, 0)

    if ibm_maxiter <= 0:
        electronic_energy = evaluate_energy_at_parameters(parameter_vector)
        return {
            "stage": "ibm",
            "mode": "single_evaluation",
            "electronic_energy_hartree": electronic_energy,
            "best_parameters": parameter_vector,
            "backend_name": selected_backend.name,
            "ibm_maxiter": 0,
        }

    expectation_history: list[float] = []

    def objective_hardware_expectation(theta_list: list[float]) -> float:
        energy = evaluate_energy_at_parameters(theta_list)
        expectation_history.append(energy)
        return energy

    optimization_result = minimize(
        objective_hardware_expectation,
        parameter_vector,
        method="COBYLA",
        options={"maxiter": ibm_maxiter},
    )
    best_theta = list(np.asarray(optimization_result.x, dtype=float))
    return {
        "stage": "ibm",
        "mode": "cobyla_on_hardware",
        "best_electronic_energy_hartree": float(optimization_result.fun),
        "best_parameters": best_theta,
        "optimizer_success": bool(optimization_result.success),
        "optimizer_nfev": int(optimization_result.nfev),
        "backend_name": selected_backend.name,
        "ibm_maxiter": ibm_maxiter,
        "expectation_history_length": len(expectation_history),
    }


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвейер H₂ VQE: ideal → noisy Aer → IBM Quantum EstimatorV2.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help="Этапы: all | ideal | noisy | ibm (по умолчанию all).",
    )
    parser.add_argument(
        "--use-pyscf",
        action="store_true",
        help="Гамильтониан через PySCF (если доступен).",
    )
    parser.add_argument(
        "--maxiter-ideal",
        type=int,
        default=200,
        help="COBYLA maxiter для идеального StatevectorEstimator.",
    )
    parser.add_argument(
        "--maxiter-noisy",
        type=int,
        default=80,
        help="COBYLA maxiter для шумного Aer EstimatorV2.",
    )
    parser.add_argument(
        "--noisy-shots",
        type=int,
        default=4096,
        help="Число снимков на оценку ожидания в шумном симуляторе.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed для шумного Aer (воспроизводимость).",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="JSON с ключом best_parameters (список float) для этапа ibm без ideal/noisy.",
    )
    parser.add_argument(
        "--ibm-backend",
        type=str,
        default=None,
        help="Имя бэкенда IBM Quantum; иначе least_busy.",
    )
    parser.add_argument(
        "--ibm-optimization-level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Уровень оптимизации transpile для железа.",
    )
    parser.add_argument(
        "--ibm-maxiter",
        type=int,
        default=0,
        help="Если >0 — COBYLA на железе (дорого). По умолчанию 0: одна оценка энергии.",
    )
    return parser.parse_args()


def normalize_stage_list(stage_tokens: list[str]) -> list[str]:
    if "all" in [token.lower() for token in stage_tokens]:
        return ["ideal", "noisy", "ibm"]
    normalized: list[str] = []
    for token in stage_tokens:
        lowered = token.lower()
        if lowered not in {"ideal", "noisy", "ibm"}:
            raise ValueError(f"Неизвестный этап: {token}")
        normalized.append(lowered)
    return normalized


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    configure_utf8_stdout()
    command_line_arguments = parse_command_line()
    project_root = Path(__file__).resolve().parent
    output_directory = ensure_output_directory(project_root)

    stage_list = normalize_stage_list(command_line_arguments.stages)

    qubit_hamiltonian, nuclear_repulsion_energy, hamiltonian_label = load_qubit_hamiltonian(
        command_line_arguments.use_pyscf,
    )
    exact_electronic_energy = exact_min_electronic_energy_hartree(qubit_hamiltonian)

    print(f"Источник гамильтониана: {hamiltonian_label}")
    print(f"Точная электронная энергия (диагонализация): {exact_electronic_energy:.8f} Ha")
    print(f"Ядерное отталкивание E_nuc: {nuclear_repulsion_energy:.8f} Ha")

    ansatz_circuit = build_variational_ansatz_h2()
    ansatz_circuit.name = "H2_STO3G_JW_HF_RealAmplitudes"

    pipeline_summary: dict = {
        "hamiltonian_source": hamiltonian_label,
        "exact_electronic_energy_hartree": exact_electronic_energy,
        "nuclear_repulsion_energy_hartree": nuclear_repulsion_energy,
        "stages_requested": stage_list,
    }

    best_parameters_for_next_stage: list[float] | None = None

    if "ideal" in stage_list:
        ideal_result = run_stage_ideal_statevector(
            ansatz_circuit,
            qubit_hamiltonian,
            nuclear_repulsion_energy,
            exact_electronic_energy,
            command_line_arguments.maxiter_ideal,
            output_directory,
        )
        pipeline_summary["ideal"] = ideal_result
        best_parameters_for_next_stage = ideal_result["best_parameters"]
        print(
            f"[ideal] ⟨H_el⟩ min = {ideal_result['best_electronic_energy_hartree']:.8f} Ha, "
            f"nfev={ideal_result['optimizer_nfev']}",
        )
        print(f"[ideal] график: {ideal_result['convergence_png']}")

    if "noisy" in stage_list:
        initial_for_noisy = best_parameters_for_next_stage
        noisy_result = run_stage_noisy_aer(
            ansatz_circuit,
            qubit_hamiltonian,
            nuclear_repulsion_energy,
            exact_electronic_energy,
            command_line_arguments.maxiter_noisy,
            command_line_arguments.noisy_shots,
            command_line_arguments.seed,
            initial_for_noisy,
            output_directory,
        )
        pipeline_summary["noisy"] = noisy_result
        best_parameters_for_next_stage = noisy_result["best_parameters"]
        print(
            f"[noisy] ⟨H_el⟩ min = {noisy_result['best_electronic_energy_hartree']:.8f} Ha, "
            f"nfev={noisy_result['optimizer_nfev']}, shots={noisy_result['noisy_shots']}",
        )
        print(f"[noisy] график: {noisy_result['convergence_png']}")

    if "ibm" in stage_list:
        if command_line_arguments.params_json is not None:
            params_payload = json.loads(command_line_arguments.params_json.read_text(encoding="utf-8"))
            theta = list(params_payload["best_parameters"])
        elif best_parameters_for_next_stage is not None:
            theta = best_parameters_for_next_stage
        else:
            print(
                "Для этапа ibm без ideal/noisy укажите --params-json с best_parameters.",
            )
            sys.exit(1)
        ibm_result = run_stage_ibm_quantum(
            ansatz_circuit,
            qubit_hamiltonian,
            theta,
            command_line_arguments.ibm_backend,
            command_line_arguments.ibm_optimization_level,
            command_line_arguments.ibm_maxiter,
        )
        pipeline_summary["ibm"] = ibm_result
        if ibm_result.get("mode") == "single_evaluation":
            print(
                f"[ibm] ⟨H_el⟩ ≈ {ibm_result['electronic_energy_hartree']:.8f} Ha "
                f"на {ibm_result['backend_name']}",
            )
        else:
            print(
                f"[ibm] ⟨H_el⟩ min ≈ {ibm_result['best_electronic_energy_hartree']:.8f} Ha "
                f"на {ibm_result['backend_name']}, nfev={ibm_result['optimizer_nfev']}",
            )

    summary_path = output_directory / "vqe_h2_pipeline_summary.json"
    if best_parameters_for_next_stage is not None:
        pipeline_summary["best_parameters_final"] = best_parameters_for_next_stage
        best_params_path = output_directory / "vqe_h2_pipeline_best_params.json"
        best_params_path.write_text(
            json.dumps({"best_parameters": best_parameters_for_next_stage}, indent=2),
            encoding="utf-8",
        )
        print(f"Параметры для --params-json: {best_params_path}")
    summary_path.write_text(
        json.dumps(pipeline_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Сводка JSON: {summary_path}")


if __name__ == "__main__":
    main()
