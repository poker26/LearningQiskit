"""
Загружает QISKIT_IBM_TOKEN из .env, сохраняет учётные данные для IBM Quantum
и проверяет доступ к облаку (список бэкендов).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService


def load_project_environment_file() -> None:
    project_root = Path(__file__).resolve().parent
    environment_file_path = project_root / ".env"
    load_dotenv(dotenv_path=environment_file_path)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_project_environment_file()

    api_token = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
    if not api_token:
        print(
            "Не найден QISKIT_IBM_TOKEN. Создайте файл .env рядом со скриптом "
            "и добавьте строку QISKIT_IBM_TOKEN=ваш_ключ."
        )
        raise SystemExit(1)

    quantum_channel = "ibm_quantum_platform"

    QiskitRuntimeService.save_account(
        channel=quantum_channel,
        token=api_token,
        overwrite=True,
        set_as_default=True,
    )
    print(
        "Учётные данные IBM Quantum сохранены для канала "
        f"{quantum_channel} (по умолчанию)."
    )

    runtime_service = QiskitRuntimeService(channel=quantum_channel)
    available_backends = runtime_service.backends()
    operational_backends = [backend for backend in available_backends if backend.status().operational]

    print(f"Доступно бэкендов: {len(available_backends)}, операционных: {len(operational_backends)}.")
    preview_limit = 12
    for backend in operational_backends[:preview_limit]:
        print(f"  - {backend.name}")
    if len(operational_backends) > preview_limit:
        remaining_count = len(operational_backends) - preview_limit
        print(f"  … и ещё {remaining_count}.")


if __name__ == "__main__":
    main()
