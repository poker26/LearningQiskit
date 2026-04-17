"""
Рисует эволюцию вероятностей базисных состояний в 2-кубитном Гровере (метка |11⟩).
Сохраняет PNG в output/ — удобно для комикса или слайдов.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from grover_two_qubits import append_two_qubit_diffuser


def build_grover_prefix(*, with_oracle: bool, with_diffuser: bool) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h([0, 1])
    if with_oracle:
        circuit.cz(0, 1)
    if with_diffuser:
        append_two_qubit_diffuser(circuit)
    return circuit


def basis_probability_vector(state_vector: Statevector) -> tuple[list[str], np.ndarray]:
    basis_labels = ["00", "01", "10", "11"]
    probability_by_label = state_vector.probabilities_dict()
    probability_array = np.array([float(probability_by_label.get(label, 0.0)) for label in basis_labels])
    return basis_labels, probability_array


def configure_matplotlib_cyrillic() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    configure_matplotlib_cyrillic()

    project_root = Path(__file__).resolve().parent
    output_directory = project_root / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    stages: list[tuple[str, Statevector]] = [
        (
            "1. После H ⊗ H\n(равномерная суперпозиция)",
            Statevector(build_grover_prefix(with_oracle=False, with_diffuser=False)),
        ),
        (
            "2. После оракула CZ\n(фаза −1 у |11⟩)",
            Statevector(build_grover_prefix(with_oracle=True, with_diffuser=False)),
        ),
        (
            "3. После диффузии\n(усиление |11⟩)",
            Statevector(build_grover_prefix(with_oracle=True, with_diffuser=True)),
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    figure.suptitle(
        "Гровер на 2 кубитах: как меняются вероятности |00⟩…|11⟩",
        fontsize=14,
        fontweight="bold",
    )

    bar_positions = np.arange(4)
    for axis, (stage_title, state_vector) in zip(axes, stages, strict=True):
        labels, probabilities = basis_probability_vector(state_vector)
        bars = axis.bar(bar_positions, probabilities, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        axis.set_xticks(bar_positions)
        axis.set_xticklabels([f"|{label}⟩" for label in labels])
        axis.set_ylim(0, 1.05)
        axis.set_title(stage_title, fontsize=10)
        axis.set_ylabel("Вероятность при измерении")
        for bar, probability_value in zip(bars, probabilities, strict=True):
            if probability_value > 0.02:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{probability_value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    figure.tight_layout()
    output_path = output_directory / "grover_probability_steps.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Сохранено: {output_path}")

    _draw_geometric_sketch(output_directory / "grover_geometry_sketch.png")


def _draw_geometric_sketch(output_path: Path) -> None:
    """Схематичный «поворот» в плоскости Гровера: |ω⟩ и |s'⟩."""
    configure_matplotlib_cyrillic()
    figure, axis = plt.subplots(figsize=(6, 5))

    theta = np.linspace(0, 2 * np.pi, 200)
    axis.plot(np.cos(theta), np.sin(theta), color="#cccccc", linestyle="--", linewidth=1)
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axvline(0, color="#999999", linewidth=0.8)
    axis.set_aspect("equal")
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.35, 1.35)
    axis.set_title("Наглядно: одна итерация — поворот ближе к «метке»", fontsize=12, fontweight="bold")

    # |s'⟩ — равномерная суперпозиция непомеченных (2D подпространство), |ω⟩ — помеченное
    angle_start = np.deg2rad(45)
    angle_oracle = np.deg2rad(90)
    axis.annotate(
        "",
        xy=(np.cos(angle_oracle), np.sin(angle_oracle)),
        xytext=(np.cos(angle_start), np.sin(angle_start)),
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2),
    )
    axis.text(0.72, 0.92, "Оракул:\nфаза метки", fontsize=9, color="#C44E52")

    angle_after_diffuser = np.deg2rad(0)
    axis.annotate(
        "",
        xy=(np.cos(angle_after_diffuser), np.sin(angle_after_diffuser)),
        xytext=(np.cos(angle_oracle), np.sin(angle_oracle)),
        arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=2),
    )
    axis.text(1.05, 0.05, "Диффузия:\nотражение", fontsize=9, color="#4C72B0")

    axis.plot([0, np.cos(angle_start)], [0, np.sin(angle_start)], color="#333333", lw=1.5)
    axis.text(0.55, 0.55, "|s⟩ старт", fontsize=10)
    axis.plot([0, 1.0], [0, 0], color="#8172B2", lw=2)
    axis.text(1.08, -0.12, "|ω⟩ метка\n(напр. |11⟩)", fontsize=10, color="#8172B2")

    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(
        0,
        -1.2,
        "Угол поворота за пару шагов ~2/√N; для N=4 одна итерация почти достаточна.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    main()
