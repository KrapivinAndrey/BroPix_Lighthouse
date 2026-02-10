from typing import List, Tuple

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

