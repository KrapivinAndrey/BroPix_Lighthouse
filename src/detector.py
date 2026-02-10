"""
Модуль детекции людей на кадре с помощью модели YOLO.

Сфокусирован только на классе "person" (люди) из набора COCO.
Для работы требуется установленный пакет `ultralytics` и файл модели,
поддерживаемый этим пакетом (например, `yolo11n.pt` или `yolov8n.pt`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
        self._model = None
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
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Выполнить детекцию людей на одном кадре.

        Args:
            frame: Изображение в формате BGR (как возвращает OpenCV).

        Returns:
            Список прямоугольников с людьми:
            [(x1, y1, x2, y2, score), ...],
            где (x1, y1) — левый верхний угол, (x2, y2) — правый нижний,
            score — уверенность модели (0..1).
        """
        if self._model is None:
            # Модель не загружена — детекция отключена
            return []

        try:
            # Ultralytics YOLO умеет принимать кадр напрямую (BGR/RGB),
            # здесь дополнительная конвертация не требуется.
            results = self._model.predict(
                frame,
                imgsz=self.config.imgsz,
                conf=self.config.conf,
                device=self.config.device or "cpu",
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка во время инференса YOLO: %s", exc)
            return []

        detections: List[Tuple[int, int, int, int, float]] = []

        # Разбираем результаты детекции, фильтруем только класс PERSON_CLASS_ID
        for result in results:
            # result.boxes.xyxy: Tensor[N, 4]
            # result.boxes.cls: Tensor[N]
            # result.boxes.conf: Tensor[N]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = getattr(boxes, "xyxy", None)
            classes = getattr(boxes, "cls", None)
            scores = getattr(boxes, "conf", None)

            if xyxy is None or classes is None or scores is None:
                continue

            for box, cls_id, score in zip(xyxy, classes, scores):
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

                detections.append((x1, y1, x2, y2, score_float))

        return detections

