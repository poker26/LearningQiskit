"""
Электронный гамильтониан H₂ в базисе STO-3G, равновесная длина связи 0.735 Å.
Jordan–Wigner, 4 кубита — совпадает с выводом PySCFDriver по умолчанию в Qiskit Nature
(см. tutorials/06_qubit_mappers.ipynb, ячейка с print(qubit_jw_op)).

Нужен для Windows и прочих сред без компилятора PySCF. На Linux при желании можно
пересчитать через PySCF и сверить.
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

# Ядерное отталкивание (Hartree) — то же, что в tutorial 01_electronic_structure Nature ~0.71997
H2_STO3G_EQUILIBRIUM_NUCLEAR_REPULSION_HARTREE = 0.719968994449

# Коэффициенты из вывода Qiskit Nature (PySCF, H2, sto3g, JW)
_H2_JW_LABELS = [
    "IIII",
    "IIIZ",
    "IIZI",
    "IIZZ",
    "IZII",
    "IZIZ",
    "YYYY",
    "XXYY",
    "YYXX",
    "XXXX",
    "ZIII",
    "ZIIZ",
    "IZZI",
    "ZIZI",
    "ZZII",
]
_H2_JW_COEFFS = [
    -0.8105479805373283,
    0.17218393261915543,
    -0.2257534922240237,
    0.12091263261776633,
    0.17218393261915543,
    0.16892753870087907,
    0.045232799946057826,
    0.045232799946057826,
    0.045232799946057826,
    0.045232799946057826,
    -0.22575349222402363,
    0.1661454325638241,
    0.16614543256382408,
    0.1746434306830045,
    0.12091263261776633,
]


def build_h2_sto3g_jordan_wigner_hamiltonian_builtin() -> SparsePauliOp:
    return SparsePauliOp.from_list(list(zip(_H2_JW_LABELS, _H2_JW_COEFFS, strict=True)))
