from __future__ import annotations

"""
Инициализация и миграции SQLite базы данных.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger(__name__)


def init_database(db_path: Path) -> None:
    """
    Инициализировать SQLite базу данных с необходимой схемой.

    Создает таблицу для событий превышения скорости и индексы для быстрого поиска.

    Args:
        db_path: Путь к файлу базы данных SQLite
    """
    # Создаем директорию, если её нет
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Создаем таблицу событий превышения скорости
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS speed_exceedance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                track_id INTEGER,
                speed_kmh REAL NOT NULL,
                speed_limit_kmh REAL NOT NULL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                detection_score REAL
            )
            """
        )

        # Создаем индекс по timestamp для быстрого поиска по времени
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON speed_exceedance_events(timestamp)
            """
        )

        # Создаем индекс по track_id для группировки событий по объектам
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_track_id 
            ON speed_exceedance_events(track_id)
            """
        )

        conn.commit()
        logger.info(f"База данных инициализирована: {db_path}")

    finally:
        conn.close()


def migrate_database(db_path: Path) -> None:
    """
    Выполнить миграции базы данных (для будущих обновлений схемы).

    Args:
        db_path: Путь к файлу базы данных SQLite
    """
    # В будущем здесь можно добавить логику миграций
    # Например, проверка версии схемы и применение обновлений
    pass
