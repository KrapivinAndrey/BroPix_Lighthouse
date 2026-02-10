from typing import Optional, Tuple

import cv2
import numpy as np
import pytest

from src.camera import Camera, CameraError


class DummyCapture:
    """Простая заглушка вместо cv2.VideoCapture для тестов."""

    def __init__(self, should_open: bool = True) -> None:
        self._opened = should_open
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - совместимость с OpenCV API
        return self._opened

    def set(self, prop_id: int, value: float) -> None:  # noqa: D401
        """Сохраняем установленные свойства, чтобы можно было проверить, что их вызывали."""
        self.props[prop_id] = value

    def get(self, prop_id: int) -> float:
        # Возвращаем разумные значения по умолчанию
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Возвращаем «пустой» чёрный кадр
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self._opened = False


def test_camera_connect_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Успешное подключение камеры при isOpened == True."""

    def fake_videocapture(index: int) -> DummyCapture:  # noqa: ARG001 - index не нужен в тесте
        return DummyCapture(should_open=True)

    monkeypatch.setattr("src.camera.cv2.VideoCapture", fake_videocapture)

    cam = Camera(index=0, width=800, height=600, fps=25)

    assert cam.connect() is True
    assert cam._is_connected is True  # noqa: SLF001 - проверяем внутреннее состояние для теста
    assert cam.cap is not None


def test_camera_connect_fails_when_not_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если камера не открылась, должен быть CameraError."""

    def fake_videocapture(index: int) -> DummyCapture:  # noqa: ARG001
        return DummyCapture(should_open=False)

    monkeypatch.setattr("src.camera.cv2.VideoCapture", fake_videocapture)

    cam = Camera(index=1)

    with pytest.raises(CameraError) as exc_info:
        cam.connect()

    assert "Failed to open camera" in str(exc_info.value)


def test_camera_read_frame_without_connect_raises() -> None:
    """Вызов read_frame без connect должен поднимать CameraError."""
    cam = Camera(index=0)

    with pytest.raises(CameraError):
        cam.read_frame()


def test_camera_read_frame_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """После успешного connect read_frame возвращает кадр."""

    def fake_videocapture(index: int) -> DummyCapture:  # noqa: ARG001
        return DummyCapture(should_open=True)

    monkeypatch.setattr("src.camera.cv2.VideoCapture", fake_videocapture)

    cam = Camera(index=0)
    cam.connect()

    ret, frame = cam.read_frame()

    assert ret is True
    assert isinstance(frame, np.ndarray)


def test_camera_release_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """release можно вызывать несколько раз без ошибок."""

    def fake_videocapture(index: int) -> DummyCapture:  # noqa: ARG001
        return DummyCapture(should_open=True)

    monkeypatch.setattr("src.camera.cv2.VideoCapture", fake_videocapture)

    cam = Camera(index=0)
    cam.connect()

    cam.release()
    # Повторный вызов не должен падать
    cam.release()


def test_camera_connect_unexpected_error_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если при создании VideoCapture выбрасывается ошибка, она оборачивается в CameraError."""

    def failing_videocapture(index: int):  # noqa: ARG001, ANN001
        raise RuntimeError("low-level failure")

    monkeypatch.setattr("src.camera.cv2.VideoCapture", failing_videocapture)

    cam = Camera(index=0)

    with pytest.raises(CameraError) as exc_info:
        cam.connect()

    assert "Unexpected error connecting to camera" in str(exc_info.value)


def test_camera_context_manager_calls_connect_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверяем, что контекстный менеджер вызывает connect и release."""

    calls = {"connect": 0, "release": 0}

    def fake_connect(self: Camera) -> bool:  # noqa: D401
        """Счётчик вызовов connect."""
        calls["connect"] += 1
        return True

    def fake_release(self: Camera) -> None:  # noqa: D401
        """Счётчик вызовов release."""
        calls["release"] += 1
        # Не вызываем оригинальный release, чтобы не трогать реальные ресурсы.

    monkeypatch.setattr("src.camera.Camera.connect", fake_connect)
    monkeypatch.setattr("src.camera.Camera.release", fake_release)

    with Camera(index=0) as cam:
        assert isinstance(cam, Camera)

    assert calls["connect"] == 1
    assert calls["release"] == 1
