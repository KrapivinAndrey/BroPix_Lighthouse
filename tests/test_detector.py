import builtins
import importlib
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

import src.detector as detector_module
from src.detector import DetectionConfig


class DummyYOLO:
    """Простая заглушка вместо ultralytics.YOLO."""

    def __init__(self, model_path: str) -> None:  # noqa: D401
        """Сохраняем путь к модели для проверки в тестах."""
        self.model_path = model_path

    def predict(self, frame: np.ndarray, *args, **kwargs) -> List[object]:  # noqa: D401
        """Возвращаем фиктивный список результатов с одним человеком и одним не-человеком."""
        # Один человек (cls = 0) и один объект другого класса (cls = 1)
        boxes = SimpleNamespace(
            xyxy=[
                [10.0, 20.0, 110.0, 220.0],
                [5.0, 5.0, 15.0, 15.0],
            ],
            cls=[0, 1],
            conf=[0.9, 0.8],
        )
        result = SimpleNamespace(boxes=boxes)
        return [result]


def test_yolo_people_detector_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если ultralytics недоступен, is_available должен вернуть False и detect() — пустой список."""

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "ultralytics":
            raise ImportError("no ultralytics")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Перезагружаем модуль, чтобы сработал новый импорт ultralytics
    importlib.reload(detector_module)

    det = detector_module.YOLOPeopleDetector()

    assert det.is_available() is False
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_yolo_people_detector_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """При успешной загрузке модели detect() должен возвращать только людей."""

    # Подменяем импорт ultralytics.YOLO на DummyYOLO
    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "ultralytics":
            return SimpleNamespace(YOLO=DummyYOLO)
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Перезагружаем модуль, чтобы использовался подменённый импорт
    importlib.reload(detector_module)

    cfg = DetectionConfig(model_path="dummy.pt", conf=0.4, imgsz=320)
    det = detector_module.YOLOPeopleDetector(cfg)

    assert det.is_available() is True

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = det.detect(frame)

    # Ожидаем один детект для класса person (cls = 0)
    assert len(detections) == 1
    x1, y1, x2, y2, score, track_id = detections[0]
    assert (x1, y1, x2, y2) == (10, 20, 110, 220)
    assert score == pytest.approx(0.9, rel=1e-3)
    # В режиме без реального трекинга track_id может быть None.
    assert track_id is None or isinstance(track_id, int)
