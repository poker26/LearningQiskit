"""
Рисует эволюцию вероятностей базисных состояний в 2-кубитном Гровере (метка |11⟩).
Дополнительно: амплитуды и «инверсия относительно среднего» для шага диффузии.
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


def computational_basis_amplitudes_real(state_vector: Statevector) -> np.ndarray:
    """Амплитуды в порядке |00⟩,|01⟩,|10⟩,|11⟩; для нашей схемы достаточно вещественной части."""
    complex_amplitudes = np.asarray(state_vector.data, dtype=np.complex128)
    if complex_amplitudes.shape != (4,):
        raise ValueError("Ожидается 2 кубита (4 амплитуды).")
    return np.real(complex_amplitudes)


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

    _draw_geometric_sketch_from_statevectors(
        [stage_sv for _, stage_sv in stages],
        output_directory / "grover_geometry_sketch.png",
    )

    _draw_diffusion_amplitude_story(
        output_directory / "grover_diffusion_amplitudes.png",
    )


def _draw_diffusion_amplitude_story(output_path: Path) -> None:
    """
    Три кадра: равные амплитуды; оракул (знак у метки); диффузия как отражение относительно ā после оракула.
    """
    configure_matplotlib_cyrillic()

    state_after_hadamards = Statevector(build_grover_prefix(with_oracle=False, with_diffuser=False))
    state_after_oracle = Statevector(build_grover_prefix(with_oracle=True, with_diffuser=False))
    state_after_diffusion = Statevector(build_grover_prefix(with_oracle=True, with_diffuser=True))

    amplitudes_uniform = computational_basis_amplitudes_real(state_after_hadamards)
    amplitudes_oracle = computational_basis_amplitudes_real(state_after_oracle)
    amplitudes_diffused = computational_basis_amplitudes_real(state_after_diffusion)

    mean_after_uniform = float(np.mean(amplitudes_uniform))
    mean_after_oracle = float(np.mean(amplitudes_oracle))
    amplitudes_predicted_by_inversion = 2.0 * mean_after_oracle - amplitudes_oracle
    max_residual = float(np.max(np.abs(amplitudes_diffused - amplitudes_predicted_by_inversion)))

    basis_tick_labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    bar_index_array = np.arange(4)
    unmarked_bar_color = "#4C72B0"
    marked_bar_color = "#8172B2"

    def bar_colors_for_amplitudes(amplitude_array: np.ndarray) -> list[str]:
        return [
            marked_bar_color if basis_index == 3 else unmarked_bar_color
            for basis_index in range(amplitude_array.shape[0])
        ]

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    figure.suptitle(
        "Диффузия в Гровере (2 кубита): амплитуды и инверсия относительно среднего после оракула",
        fontsize=13,
        fontweight="bold",
    )

    axis_uniform, axis_oracle, axis_diffusion = axes

    bar_container_uniform = axis_uniform.bar(
        bar_index_array,
        amplitudes_uniform,
        color=bar_colors_for_amplitudes(amplitudes_uniform),
        edgecolor="white",
        linewidth=0.8,
    )
    axis_uniform.axhline(
        mean_after_uniform,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=rf"$\bar a = {mean_after_uniform:.2f}$",
    )
    axis_uniform.set_title("Шаг 1: после $H^{\\otimes 2}$\nвсе амплитуды равны", fontsize=11)
    axis_uniform.set_ylabel("Амплитуда (Re)")
    axis_uniform.set_xticks(bar_index_array)
    axis_uniform.set_xticklabels(basis_tick_labels)
    axis_uniform.legend(loc="upper right", fontsize=9)
    axis_uniform.set_ylim(0, 0.62)
    for bar_handle, amplitude_value in zip(bar_container_uniform, amplitudes_uniform, strict=True):
        axis_uniform.text(
            bar_handle.get_x() + bar_handle.get_width() / 2,
            bar_handle.get_height() + 0.02,
            f"{amplitude_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bar_container_oracle = axis_oracle.bar(
        bar_index_array,
        amplitudes_oracle,
        color=bar_colors_for_amplitudes(amplitudes_oracle),
        edgecolor="white",
        linewidth=0.8,
    )
    axis_oracle.axhline(
        mean_after_oracle,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=rf"новое $\bar a = {mean_after_oracle:.2f}$",
    )
    axis_oracle.set_title("Шаг 2: оракул $CZ$\nфаза $-1$ у $|11\\rangle$", fontsize=11)
    axis_oracle.set_xticks(bar_index_array)
    axis_oracle.set_xticklabels(basis_tick_labels)
    axis_oracle.legend(loc="upper right", fontsize=9)
    axis_oracle.axhline(0.0, color="#bbbbbb", linewidth=0.9)
    axis_oracle.set_ylim(-0.62, 0.62)
    for bar_handle, amplitude_value in zip(bar_container_oracle, amplitudes_oracle, strict=True):
        vertical_offset = 0.03 if amplitude_value >= 0 else -0.08
        vertical_alignment = "bottom" if amplitude_value >= 0 else "top"
        axis_oracle.text(
            bar_handle.get_x() + bar_handle.get_width() / 2,
            amplitude_value + vertical_offset,
            f"{amplitude_value:.2f}",
            ha="center",
            va=vertical_alignment,
            fontsize=9,
        )

    grouped_bar_width = 0.36
    _ = axis_diffusion.bar(
        bar_index_array - grouped_bar_width / 2,
        amplitudes_oracle,
        width=grouped_bar_width,
        color=bar_colors_for_amplitudes(amplitudes_oracle),
        alpha=0.45,
        edgecolor="white",
        linewidth=0.6,
        label="после оракула",
    )
    bar_container_after_diffuser = axis_diffusion.bar(
        bar_index_array + grouped_bar_width / 2,
        amplitudes_diffused,
        width=grouped_bar_width,
        color=bar_colors_for_amplitudes(amplitudes_diffused),
        alpha=1.0,
        edgecolor="white",
        linewidth=0.6,
        label="после диффузии",
    )
    axis_diffusion.axhline(
        mean_after_oracle,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=rf"ось отражения: $\bar a={mean_after_oracle:.2f}$",
    )
    axis_diffusion.set_title(
        "Шаг 3: диффузия\n$a_i \\to 2\\bar a - a_i$ (это и есть диффузия)",
        fontsize=11,
    )
    axis_diffusion.set_xticks(bar_index_array)
    axis_diffusion.set_xticklabels(basis_tick_labels)
    axis_diffusion.legend(loc="upper left", fontsize=8)
    axis_diffusion.axhline(0.0, color="#bbbbbb", linewidth=0.9)
    axis_diffusion.set_ylim(-1.08, 1.08)

    marked_basis_index = 3
    axis_diffusion.annotate(
        "",
        xy=(
            marked_basis_index + grouped_bar_width / 2,
            amplitudes_diffused[marked_basis_index],
        ),
        xytext=(
            marked_basis_index - grouped_bar_width / 2,
            amplitudes_oracle[marked_basis_index],
        ),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8, shrinkA=4, shrinkB=4),
    )
    axis_diffusion.text(
        2.55,
        0.35,
        "отражение\nотносительно $\\bar a$",
        fontsize=9,
        color="#222222",
    )

    figure.text(
        0.5,
        0.02,
        (
            "Вероятности на шагах 1 и 2 совпадают ($|a|^2$), меняется фаза у метки. "
            "Диффузия использует $\\bar a$ после оракула. "
            f"Проверка $a' = 2\\bar a - a$: max $|\\Delta| \\approx {max_residual:.1e}$."
        ),
        ha="center",
        fontsize=8,
        style="italic",
    )

    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Сохранено: {output_path}")


def grover_plane_coordinates_amplitude_s_prime_omega(state_vector: Statevector) -> tuple[float, float]:
    """
    Ортонормированный базис 2D-подпространства Гровера для одной метки |11⟩:
    |ω⟩ = |11⟩, |s'⟩ = (|00⟩+|01⟩+|10⟩)/√3.
    Возвращает (⟨s'|ψ⟩, ⟨ω|ψ⟩) — вещественные части (для нашей схемы мнимая часть ~0).
    """
    amplitudes = np.asarray(state_vector.data, dtype=np.complex128)
    if amplitudes.shape != (4,):
        raise ValueError("Ожидается 2 кубита (4 амплитуды).")
    amplitude_on_unmarked_uniform = (amplitudes[0] + amplitudes[1] + amplitudes[2]) / np.sqrt(3.0)
    amplitude_on_marked = amplitudes[3]
    return float(np.real(amplitude_on_unmarked_uniform)), float(np.real(amplitude_on_marked))


def _draw_geometric_sketch_from_statevectors(
    stage_statevectors: list[Statevector],
    output_path: Path,
) -> None:
    """Стрелки по реальным проекциям |ψ⟩ на плоскость span{|s'⟩, |ω⟩}."""
    configure_matplotlib_cyrillic()
    figure, axis = plt.subplots(figsize=(7.2, 6.0))

    theta = np.linspace(0, 2 * np.pi, 240)
    axis.plot(np.cos(theta), np.sin(theta), color="#dddddd", linestyle="--", linewidth=1.2, zorder=0)
    axis.axhline(0, color="#bbbbbb", linewidth=0.9, zorder=0)
    axis.axvline(0, color="#bbbbbb", linewidth=0.9, zorder=0)
    axis.set_aspect("equal")

    axis.set_xlabel("⟨s'|ψ⟩ — непомеченные (|00⟩,|01⟩,|10⟩) равномерно", fontsize=10)
    axis.set_ylabel("⟨ω|ψ⟩,  ω = |11⟩ — метка", fontsize=10)
    axis.set_title(
        "Гровер (2 кубита): траектория в плоскости |s'⟩–|ω⟩ (из Statevector)",
        fontsize=12,
        fontweight="bold",
    )

    plane_points: list[tuple[float, float]] = []
    for stage_vector in stage_statevectors:
        coordinate_s_prime, coordinate_omega = grover_plane_coordinates_amplitude_s_prime_omega(
            stage_vector
        )
        plane_points.append((coordinate_s_prime, coordinate_omega))

    stage_labels = [
        "1) после H⊗H\n(|s⟩)",
        "2) после CZ",
        "3) после диффузии\n(−|11⟩, та же разметка)",
    ]
    colors = ["#333333", "#C44E52", "#4C72B0"]
    for stage_index, ((coordinate_x, coordinate_y), stage_label, point_color) in enumerate(
        zip(plane_points, stage_labels, colors, strict=True)
    ):
        axis.scatter(
            [coordinate_x],
            [coordinate_y],
            s=70,
            color=point_color,
            zorder=3,
            edgecolors="white",
            linewidths=1.2,
        )
        offset_x = 0.06 if stage_index != 2 else -0.22
        offset_y = 0.05 if stage_index != 1 else -0.12
        axis.annotate(
            f"{stage_label}\n({coordinate_x:.3f}, {coordinate_y:.3f})",
            (coordinate_x, coordinate_y),
            textcoords="offset points",
            xytext=(20 + offset_x * 80, 12 + offset_y * 80),
            fontsize=8,
            color=point_color,
        )

    for segment_index in range(len(plane_points) - 1):
        start_x, start_y = plane_points[segment_index]
        end_x, end_y = plane_points[segment_index + 1]
        arrow_color = "#C44E52" if segment_index == 0 else "#4C72B0"
        axis.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=2.2,
                shrinkA=6,
                shrinkB=6,
            ),
            zorder=2,
        )
        caption = "оракул (фаза |11⟩)" if segment_index == 0 else "диффузия"
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        axis.text(
            mid_x + 0.06,
            mid_y + 0.06,
            caption,
            fontsize=9,
            color=arrow_color,
        )

    # Подсказка: ось x — коэффициент при |s'⟩, ось y — при |ω⟩
    axis.plot([0, 1.0], [0, 0], color="#55A868", lw=2, alpha=0.35)
    axis.text(1.02, -0.06, "|s'⟩", fontsize=11, color="#55A868")
    axis.plot([0, 0], [0, 1.0], color="#8172B2", lw=2, alpha=0.35)
    axis.text(-0.06, 1.02, "|ω⟩", fontsize=11, color="#8172B2")

    norm_residual = [
        abs(np.linalg.norm(np.array(point)) - 1.0) for point in plane_points
    ]
    caption_footer = (
        "Точки лежат на единичной окружности: состояние остаётся в плоскости span{|s'⟩,|ω⟩}. "
        f"Отклонение нормы от 1: max {max(norm_residual):.2e}."
    )
    axis.text(0.0, -1.28, caption_footer, ha="center", fontsize=8, style="italic")

    margin = 0.22
    axis.set_xlim(-1.0 - margin, 1.0 + margin)
    axis.set_ylim(-1.0 - margin, 1.0 + margin)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    main()
