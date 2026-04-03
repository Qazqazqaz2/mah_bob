"""Точка входа: запуск из корня проекта (python main.py)."""

from __future__ import annotations

import sys
from pathlib import Path

# Пакет установлен в src/ — без pip install -e . добавляем его в path
_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_supervisor.main import main

if __name__ == "__main__":
    main()
