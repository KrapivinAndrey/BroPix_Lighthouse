from __future__ import annotations

"""
Сервис управления конфигурацией с потокобезопасностью и валидацией.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from src.speed_utils import DEFAULT_PX_TO_M_SCALE


class ConfigService:
    """
    Потокобезопасный сервис управления конфигурацией.

    Предоставляет методы для чтения и обновления настроек системы
    с автоматической валидацией значений и сохранением на диск.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """
        Инициализация сервиса конфигурации.

        Args:
            config_path: Путь к файлу конфигурации. Если None, используется
                        config.json в корне проекта.
        """
        if config_path is None:
            # Предполагаем, что config.json находится в корне проекта
            # (parent директория от src/)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.json"

        self._config_path = Path(config_path)
        self._lock = threading.Lock()
        self._config: Dict[str, Any] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Загрузить конфигурацию с диска."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self._config_path}")

        try:
            with self._config_path.open("r", encoding="utf-8") as f:
                self._config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {str(e)}") from e

        # Валидация обязательных полей
        if "camera" not in self._config:
            raise ValueError("Configuration file must contain 'camera' section")
        if "index" not in self._config["camera"]:
            raise ValueError("Camera configuration must contain 'index' field")

    def get_config(self) -> Dict[str, Any]:
        """
        Получить полную конфигурацию (копию).

        Returns:
            Словарь с настройками системы
        """
        with self._lock:
            # Глубокое копирование через JSON сериализацию
            copied: Dict[str, Any] = json.loads(json.dumps(self._config))
            return copied

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Обновить конфигурацию с валидацией.

        Args:
            updates: Словарь с обновлениями (может содержать вложенные секции)

        Raises:
            ValueError: Если значения не проходят валидацию
        """
        with self._lock:
            # Обновляем секцию camera
            if "camera" in updates:
                camera_updates = updates["camera"]
                if isinstance(camera_updates, dict):
                    camera_cfg = self._config.setdefault("camera", {})
                    for key in ("index", "width", "height", "fps"):
                        if key in camera_updates:
                            camera_cfg[key] = camera_updates[key]

            # Обновляем секцию detection
            if "detection" in updates:
                detection_updates = updates["detection"]
                if isinstance(detection_updates, dict):
                    detection_cfg = self._config.setdefault("detection", {})
                    for key in ("enabled", "model_path", "conf", "device", "imgsz", "draw_boxes"):
                        if key in detection_updates:
                            detection_cfg[key] = detection_updates[key]

            # Обновляем отдельные поля верхнего уровня
            if "speed_limit_kmh" in updates:
                try:
                    self._config["speed_limit_kmh"] = float(updates["speed_limit_kmh"])
                except (TypeError, ValueError) as e:
                    raise ValueError("speed_limit_kmh must be a number") from e

            if "red_hold_seconds" in updates:
                try:
                    red_hold = float(updates["red_hold_seconds"])
                    # Валидация диапазона 1-30 секунд (по PRD)
                    if red_hold < 1.0 or red_hold > 30.0:
                        raise ValueError("red_hold_seconds must be between 1 and 30")
                    self._config["red_hold_seconds"] = red_hold
                except (TypeError, ValueError) as e:
                    raise ValueError("red_hold_seconds must be a number between 1 and 30") from e

            if "px_to_m_scale" in updates:
                try:
                    self._config["px_to_m_scale"] = float(updates["px_to_m_scale"])
                except (TypeError, ValueError) as e:
                    raise ValueError("px_to_m_scale must be a number") from e

            if "detection_frame_stride" in updates:
                try:
                    stride = int(updates["detection_frame_stride"])
                    if stride <= 0:
                        stride = 1
                    self._config["detection_frame_stride"] = stride
                except (TypeError, ValueError) as e:
                    raise ValueError("detection_frame_stride must be a positive integer") from e

    def save_to_disk(self) -> None:
        """Сохранить текущую конфигурацию на диск."""
        with self._lock:
            with self._config_path.open("w", encoding="utf-8") as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)

    def get_camera_config(self) -> Dict[str, Any]:
        """Получить конфигурацию камеры."""
        with self._lock:
            camera_cfg = self._config.get("camera", {})
            if isinstance(camera_cfg, dict):
                return dict(camera_cfg)
            return {}

    def get_detection_config(self) -> Dict[str, Any]:
        """Получить конфигурацию детекции."""
        with self._lock:
            detection_cfg = self._config.get("detection", {})
            if isinstance(detection_cfg, dict):
                return dict(detection_cfg)
            return {}

    def get_speed_limit_kmh(self) -> float:
        """Получить порог скорости в км/ч."""
        with self._lock:
            return float(self._config.get("speed_limit_kmh", 8.0))

    def get_red_hold_seconds(self) -> float:
        """Получить время удержания красного сигнала в секундах."""
        with self._lock:
            return float(self._config.get("red_hold_seconds", 2.0))

    def get_px_to_m_scale(self) -> float:
        """Получить масштаб пикселей в метры."""
        with self._lock:
            return float(self._config.get("px_to_m_scale", DEFAULT_PX_TO_M_SCALE))

    def get_detection_frame_stride(self) -> int:
        """Получить шаг детекции по кадрам (1 = каждый кадр, 2 = каждый второй и т.д.)."""
        with self._lock:
            stride = self._config.get("detection_frame_stride", 1)
            return int(stride) if stride and stride > 0 else 1
