"""
Точка входа для веб-режима LighthouseForCycles.

Команда запуска после установки пакета:

    lighthouse-web

поднимает FastAPI-приложение с MJPEG-потоком и API.
"""

from __future__ import annotations

import uvicorn

from src.web_app import app


def main() -> None:
    """Запустить веб-сервер LighthouseForCycles."""
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
