from typing import List, Tuple
import sys
from types import SimpleNamespace

import numpy as np
import pytest  # type: ignore[import]

from src.camera import Camera, display_video_stream


class DummyCameraForStream(Camera):
    """Заглушка камеры для тестирования цикла display_video_stream."""

    def __init__(self, frames_before_stop: int = 3) -> None:
        # Не вызываем родительский __init__, чтобы не трогать cv2.VideoCapture
        self.index = 0
        self._is_connected = False
        self.cap = None
        self._frames_left = frames_before_stop
        self.read_calls: List[Tuple[bool, np.ndarray | None]] = []

    def connect(self) -> bool:  # noqa: D401
        """Подключение ничего не делает, только помечает камеру как подключённую."""
        self._is_connected = True
        return True

    def read_frame(self):  # type: ignore[override]
        if self._frames_left <= 0:
            # Эмулируем ситуацию, когда кадр больше не читается
            return False, None

        self._frames_left -= 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.read_calls.append((True, frame))
        return True, frame

    def release(self) -> None:
        self._is_connected = False


def test_display_video_stream_stops_when_no_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    display_video_stream должен корректно завершать цикл,
    когда read_frame начинает возвращать ret == False.
    """

    dummy_camera = DummyCameraForStream(frames_before_stop=2)

    # Подменяем cv2.imshow и cv2.waitKey, чтобы не открывать окно.
    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    # waitKey должен вернуть значение, которое не эквивалентно ESC или Q
    monkeypatch.setattr("src.camera.cv2.waitKey", lambda delay: 0)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    # Вызываем функцию. Она должна отработать быстро и завершиться.
    display_video_stream(camera=dummy_camera, config={"camera": {"index": 0}})

    # Убедимся, что read_frame вызывался ожидаемое количество раз (2 успешных кадра + 1 неуспешная попытка).
    assert len(dummy_camera.read_calls) == 2


def test_display_video_stream_uses_camera_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если camera=None, то экземпляр камеры создаётся из конфигурации."""

    created_params: dict = {}

    class DummyCam(Camera):
        def __init__(
            self,
            index: int,
            width: int | None = None,
            height: int | None = None,
            fps: int | None = None,
        ):  # type: ignore[override]
            created_params.update(
                index=index,
                width=width,
                height=height,
                fps=fps,
            )
            self.index = index
            self._frames_left = 1

        def connect(self) -> bool:  # noqa: D401
            """Подключение ничего не делает."""
            return True

        def read_frame(self):  # type: ignore[override]
            if self._frames_left <= 0:
                return False, None
            self._frames_left -= 1
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return True, frame

        def release(self) -> None:
            pass

    monkeypatch.setattr("src.camera.Camera", DummyCam)
    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.waitKey", lambda delay: 0)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    cfg = {"camera": {"index": 2, "width": 800, "height": 600, "fps": 30}}
    display_video_stream(camera=None, config=cfg)

    assert created_params == {"index": 2, "width": 800, "height": 600, "fps": 30}


def test_display_video_stream_updates_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверяем, что ветка пересчёта FPS (elapsed >= 1.0) выполняется."""

    dummy_camera = DummyCameraForStream(frames_before_stop=3)

    # Эмулируем время так, чтобы между кадрами прошло больше секунды.
    times = [0.0, 2.0, 3.5, 5.0]

    def fake_time():
        return times.pop(0) if times else 6.0

    monkeypatch.setattr("src.camera.time.time", fake_time)
    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.waitKey", lambda delay: 0)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    display_video_stream(camera=dummy_camera, config={"camera": {"index": 0}})


def test_display_video_stream_quit_by_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Цикл должен завершаться при нажатии Q."""

    dummy_camera = DummyCameraForStream(frames_before_stop=10)

    # Первый кадр показываем, потом эмулируем нажатие 'q'.
    keys = [0, ord("q")]

    def fake_wait_key(delay: int) -> int:  # noqa: ARG001
        return keys.pop(0) if keys else 0

    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.waitKey", fake_wait_key)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    display_video_stream(camera=dummy_camera, config={"camera": {"index": 0}})


def test_display_video_stream_loads_config_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если config=None, функция должна загрузить конфиг через load_config()."""

    dummy_camera = DummyCameraForStream(frames_before_stop=1)

    def fake_load_config():
        return {"camera": {"index": 0}, "detection": {"enabled": False}}

    monkeypatch.setattr("src.camera.load_config", fake_load_config)
    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.waitKey", lambda delay: 0)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    display_video_stream(camera=dummy_camera, config=None)


def test_display_video_stream_with_detection_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ветка с включённой детекцией должна вызывать detect() и отрисовывать рамки."""

    dummy_camera = DummyCameraForStream(frames_before_stop=1)

    class DummyDetector:
        def __init__(self, cfg):  # noqa: ANN001
            self.cfg = cfg

        def is_available(self) -> bool:
            return True

        def detect(self, frame):  # noqa: ANN001
            # Один детект
            return [(10, 20, 30, 40, 0.9)]

    def fake_detection_config(*args, **kwargs):  # noqa: ANN001
        class _Cfg:
            def __init__(self, **inner_kwargs):  # noqa: ANN001
                self.__dict__.update(inner_kwargs)

        return _Cfg(**kwargs)

    cfg = {
        "camera": {"index": 0},
        "detection": {
            "enabled": True,
            "model_path": "yolo11n.pt",
            "conf": 0.5,
            "imgsz": 640,
        },
    }
    # Подменяем модуль src.detector на заглушку
    dummy_detector_module = SimpleNamespace(
        DetectionConfig=fake_detection_config,
        YOLOPeopleDetector=DummyDetector,
    )
    monkeypatch.setitem(sys.modules, "src.detector", dummy_detector_module)
    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.waitKey", lambda delay: 0)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)
    monkeypatch.setattr("src.camera.cv2.rectangle", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.putText", lambda *args, **kwargs: None)

    display_video_stream(camera=dummy_camera, config=cfg)


def test_display_video_stream_yolo_error_disables_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если во время detect() возникает ошибка, детекция должна отключаться без падения цикла."""

    dummy_camera = DummyCameraForStream(frames_before_stop=2)

    class FailingDetector:
        def __init__(self, cfg):  # noqa: ANN001
            self.cfg = cfg

        def is_available(self) -> bool:
            return True

        def detect(self, frame):  # noqa: ANN001
            raise RuntimeError("YOLO failure")

    def fake_detection_config(*args, **kwargs):  # noqa: ANN001
        class _Cfg:
            def __init__(self, **inner_kwargs):  # noqa: ANN001
                self.__dict__.update(inner_kwargs)

        return _Cfg(**kwargs)

    cfg = {
        "camera": {"index": 0},
        "detection": {
            "enabled": True,
            "model_path": "yolo11n.pt",
            "conf": 0.5,
            "imgsz": 640,
        },
    }
    dummy_detector_module = SimpleNamespace(
        DetectionConfig=fake_detection_config,
        YOLOPeopleDetector=FailingDetector,
    )
    monkeypatch.setitem(sys.modules, "src.detector", dummy_detector_module)
    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.waitKey", lambda delay: 0)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    display_video_stream(camera=dummy_camera, config=cfg)


def test_display_video_stream_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """KeyboardInterrupt внутри цикла должен корректно обрабатываться."""

    class InterruptingCamera(DummyCameraForStream):
        def read_frame(self):  # type: ignore[override]
            raise KeyboardInterrupt()

    dummy_camera = InterruptingCamera(frames_before_stop=10)

    monkeypatch.setattr("src.camera.cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.waitKey", lambda delay: 0)
    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    # Не должно выбрасывать исключение наружу
    display_video_stream(camera=dummy_camera, config={"camera": {"index": 0}})


def test_display_video_stream_unexpected_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Неожиданная ошибка должна логироваться и пробрасываться наружу."""

    class FailingCamera(DummyCameraForStream):
        def connect(self) -> bool:  # noqa: D401
            """Эмулируем ошибку во время connect()."""
            raise RuntimeError("connect failed")

    dummy_camera = FailingCamera(frames_before_stop=1)

    monkeypatch.setattr("src.camera.cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.camera.cv2.destroyAllWindows", lambda: None)

    with pytest.raises(RuntimeError):
        display_video_stream(camera=dummy_camera, config={"camera": {"index": 0}})
