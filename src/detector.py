"""
Модуль детекции людей на кадре с помощью модели YOLO.

Сфокусирован только на классе "person" (люди) из набора COCO.
Для работы требуется установленный пакет `ultralytics` и файл модели,
поддерживаемый этим пакетом (например, `yolo11n.pt` или `yolov8n.pt`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """
    Конфигурация детектора YOLO.

    Attributes:
        model_path: Путь к файлу модели YOLO (например, 'yolo11n.pt').
        conf: Порог уверенности детекции (0..1).
        device: Устройство для инференса ('cpu', 'cuda', None для автоопределения).
        imgsz: Размер входного изображения для модели (одинаковая ширина/высота).
    """

    model_path: str = "yolo11n.pt"
    conf: float = 0.5
    device: Optional[str] = None
    imgsz: int = 640


class YOLOPeopleDetector:
    """
    Обертка над моделью YOLO для детекции ТОЛЬКО людей.

    Использует предобученную модель на COCO, где класс 0 обычно соответствует "person".
    """

    # В стандартной разметке COCO класс 0 — это "person"
    PERSON_CLASS_ID = 0

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Инициализация детектора людей.

        Args:
            config: Объект конфигурации детектора. Если None, используется значение по умолчанию.
        """
        self.config = config or DetectionConfig()
        # Храним модель YOLO в атрибуте с типом Any, поскольку
        # сторонняя библиотека не предоставляет стабильных аннотаций.
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Загрузка модели YOLO.

        Если библиотека `ultralytics` или файл модели недоступны, логируем предупреждение
        и продолжаем работу без детекции (detector будет возвращать пустой список).
        """
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError:
            logger.warning(
                "Библиотека 'ultralytics' не установлена. "
                "Детекция людей с помощью YOLO будет отключена."
            )
            self._model = None
            return

        try:
            logger.info(
                "Загрузка YOLO модели для детекции людей: %s",
                self.config.model_path,
            )
            self._model = YOLO(self.config.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Не удалось загрузить модель YOLO из '%s': %s. "
                "Детекция людей будет отключена.",
                self.config.model_path,
                exc,
            )
            self._model = None

    def is_available(self) -> bool:
        """Возвращает True, если модель успешно загружена и готова к работе."""
        return self._model is not None

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        """
        Выполнить детекцию людей на одном кадре.

        Args:
            frame: Изображение в формате BGR (как возвращает OpenCV).

        Returns:
            Список прямоугольников с людьми:
            [(x1, y1, x2, y2, score, track_id), ...],
            где (x1, y1) — левый верхний угол, (x2, y2) — правый нижний,
            score — уверенность модели (0..1),
            track_id — идентификатор трека (int) или None, если трекинг недоступен.
        """
        if self._model is None:
            # Модель не загружена — детекция отключена
            return []

        try:
            # Используем режим трекинга Ultralytics YOLO.
            # persist=True позволяет сохранять внутреннее состояние трекера
            # между вызовами detect().
            if hasattr(self._model, "track"):
                results = self._model.track(  # type: ignore[attr-defined]
                    frame,
                    imgsz=self.config.imgsz,
                    conf=self.config.conf,
                    device=self.config.device or "cpu",
                    verbose=False,
                    persist=True,
                )
            else:
                # Fallback для заглушек или старых версий — обычный predict без трекинга.
                results = self._model.predict(
                    frame,
                    imgsz=self.config.imgsz,
                    conf=self.config.conf,
                    device=self.config.device or "cpu",
                    verbose=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка во время инференса/трекинга YOLO: %s", exc)
            return []

        detections: List[Tuple[int, int, int, int, float, Optional[int]]] = []

        # Разбираем результаты детекции, фильтруем только класс PERSON_CLASS_ID
        for result in results:
            # result.boxes.xyxy: Tensor[N, 4]
            # result.boxes.cls: Tensor[N]
            # result.boxes.conf: Tensor[N]
            # result.boxes.id: Tensor[N] или None (для трекинга)
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = getattr(boxes, "xyxy", None)
            classes = getattr(boxes, "cls", None)
            scores = getattr(boxes, "conf", None)
            track_ids = getattr(boxes, "id", None)

            if xyxy is None or classes is None or scores is None:
                continue

            # Если трекинг не вернул id, будем подставлять None.
            if track_ids is None:
                track_ids_iter = [None] * len(xyxy)
            else:
                track_ids_iter = track_ids

            for box, cls_id, score, track_id in zip(
                xyxy,
                classes,
                scores,
                track_ids_iter,
            ):
                try:
                    cls_int = int(cls_id)
                except (TypeError, ValueError):
                    continue

                if cls_int != self.PERSON_CLASS_ID:
                    # Нас интересуют только люди
                    continue

                try:
                    x1, y1, x2, y2 = [int(float(v)) for v in box]
                    score_float = float(score)
                except (TypeError, ValueError):
                    continue

                track_int: Optional[int]
                try:
                    track_int = int(track_id) if track_id is not None else None
                except (TypeError, ValueError):
                    track_int = None

                detections.append((x1, y1, x2, y2, score_float, track_int))

        return detections
