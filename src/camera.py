"""
Camera module for USB camera capture and video streaming.

This module provides functionality to connect to USB cameras,
capture frames, and display video streams for debugging purposes.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.speed_utils import compute_speed_kmh

# Configure logging
logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Exception raised for camera-related errors."""

    pass


class Camera:
    """
    USB Camera interface using OpenCV.

    Provides methods to connect to a USB camera, capture frames,
    and manage camera resources safely.
    """

    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ):
        """
        Initialize camera with specified parameters.

        Args:
            index: Camera device index (0 for first camera)
            width: Frame width in pixels (None for default)
            height: Frame height in pixels (None for default)
            fps: Frames per second (None for default)
        """
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False

    def connect(self) -> bool:
        """
        Connect to the USB camera.

        Returns:
            True if connection successful, False otherwise

        Raises:
            CameraError: If camera cannot be opened or configured
        """
        try:
            logger.info(f"Attempting to connect to camera index {self.index}")
            self.cap = cv2.VideoCapture(self.index)

            if not self.cap.isOpened():
                raise CameraError(
                    f"Failed to open camera at index {self.index}. Camera may be in "
                    "use or not connected."
                )

            # Configure camera properties if specified
            if self.width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Camera connected successfully. Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            self._is_connected = True
            return True

        except Exception as e:
            if isinstance(e, CameraError):
                raise
            raise CameraError(f"Unexpected error connecting to camera: {str(e)}") from e

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.

        Returns:
            Tuple of (success, frame) where success is True if frame was read successfully,
            and frame is a numpy array containing the image, or None if read failed
        """
        if not self._is_connected or self.cap is None:
            raise CameraError("Camera is not connected. Call connect() first.")

        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self._is_connected = False
            logger.info("Camera resources released")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Cleanup on object destruction."""
        self.release()


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load camera configuration from JSON file.

    Args:
        config_path: Path to configuration file. If None, uses 'config.json' in project root.

    Returns:
        Dictionary containing camera configuration

    Raises:
        FileNotFoundError: If config file does not exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If camera configuration is missing or invalid
    """
    if config_path is None:
        # Assume config.json is in project root (parent of src/)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            # Явно аннотируем тип конфигурации, чтобы избежать Any из json.load.
            config: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {str(e)}") from e

    if "camera" not in config:
        raise ValueError("Configuration file must contain 'camera' section")

    camera_config = config["camera"]
    if "index" not in camera_config:
        raise ValueError("Camera configuration must contain 'index' field")

    return config


def display_video_stream(camera: Optional[Camera] = None, config: Optional[dict] = None) -> None:
    """
    Отображение видеопотока с камеры в реальном времени для отладки.

    Показывает FPS, статус камеры, а при включённой опции детекции —
    прямоугольники вокруг обнаруженных людей (класс person из COCO).
    Для выхода нажмите ESC или Q.

    Args:
        camera: Экземпляр камеры. Если None — создаётся из конфигурации.
        config: Конфигурационный словарь. Если None — загружается из config.json.
    """
    if camera is None:
        if config is None:
            config = load_config()

        camera_config = config["camera"]
        camera = Camera(
            index=camera_config.get("index", 0),
            width=camera_config.get("width"),
            height=camera_config.get("height"),
            fps=camera_config.get("fps"),
        )

    # Подготовка детектора людей YOLO (опционально, через конфиг)
    yolo_detector = None
    detection_enabled = False
    detection_config = {}

    if config is None:
        # Если конфиг не передан — пытаемся загрузить для детекции отдельно
        try:
            config = load_config()
        except Exception:
            config = None

    if config is not None:
        detection_config = config.get("detection", {})
        detection_enabled = bool(detection_config.get("enabled", False))

    if detection_enabled:
        try:
            from src.detector import DetectionConfig, YOLOPeopleDetector

            det_cfg = DetectionConfig(
                model_path=detection_config.get("model_path", "yolo11n.pt"),
                conf=float(detection_config.get("conf", 0.5)),
                device=detection_config.get("device"),
                imgsz=int(detection_config.get("imgsz", 640)),
            )
            yolo_detector = YOLOPeopleDetector(det_cfg)

            if not yolo_detector.is_available():
                logger.warning("YOLO-детектор людей недоступен, детекция будет отключена.")
                detection_enabled = False
            else:
                logger.info(
                    "YOLO-детекция людей включена (модель: %s, порог: %.2f).",
                    det_cfg.model_path,
                    det_cfg.conf,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Не удалось инициализировать YOLO-детектор людей: %s", e)
            detection_enabled = False

    try:
        camera.connect()

        window_name = "Camera Debug - Press ESC or Q to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_current = 0.0

        # Счётчик кадров видеопотока
        frame_idx = 0

        # Хранилище треков для расчёта скорости:
        # track_id -> {"last_pos": (cx, cy), "last_time": float, "speed_kmh": Optional[float]}
        tracks: Dict[int, Dict[str, Any]] = {}
        track_ttl_seconds = 2.0

        logger.info("Starting video stream. Press ESC or Q to exit.")

        while True:
            ret, frame = camera.read_frame()

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                break

            # Инкремент счётчика кадров
            frame_idx += 1

            # Расчёт FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps_current = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Детекция людей с помощью YOLO (если включена и доступна)
            if detection_enabled and yolo_detector is not None:
                try:
                    detections = yolo_detector.detect(frame)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Ошибка во время детекции людей: %s. Детекция будет отключена.", e)
                    detection_enabled = False
                    detections = []
                else:
                    now = time.time()

                    # Периодическая очистка устаревших треков
                    if frame_idx % 30 == 0:
                        expired_ids = []
                        for track_id, data in tracks.items():
                            last_time_val = data.get("last_time")
                            if isinstance(last_time_val, (int, float)) and now - last_time_val > track_ttl_seconds:
                                expired_ids.append(track_id)
                        for track_id in expired_ids:
                            tracks.pop(track_id, None)

                    # Рисуем рамки только вокруг людей и рассчитываем скорость по трек-идентификатору.
                    # Поддерживаем оба формата детекций:
                    # (x1, y1, x2, y2, score) и (x1, y1, x2, y2, score, track_id).
                    for det in detections:
                        det_track_id: Optional[int]
                        if len(det) == 5:
                            x1, y1, x2, y2, score = det
                            det_track_id = None
                        elif len(det) == 6:
                            x1, y1, x2, y2, score, det_track_id = det
                        else:
                            # Неизвестный формат — пропускаем.
                            continue
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        speed_kmh = None
                        # Порог скорости в км/ч, выше которого рамка считается «красной»
                        SPEED_LIMIT_KMH = 8.0

                        if det_track_id is not None and det_track_id >= 0:
                            prev = tracks.get(det_track_id)
                            if prev is not None:
                                last_pos = prev.get("last_pos")
                                last_time_val = prev.get("last_time")
                                if (
                                    isinstance(last_pos, tuple)
                                    and len(last_pos) == 2
                                    and isinstance(last_time_val, (int, float))
                                ):
                                    speed_kmh = compute_speed_kmh(
                                        last_pos,
                                        float(last_time_val),
                                        (cx, cy),
                                        now,
                                    )

                            tracks[det_track_id] = {
                                "last_pos": (cx, cy),
                                "last_time": now,
                                "speed_kmh": (
                                    speed_kmh
                                    if speed_kmh is not None
                                    else prev.get("speed_kmh")
                                    if prev
                                    else None
                                ),
                            }
                        else:
                            det_track_id = None

                        # Получаем последнюю оценённую скорость из трека (если есть)
                        if det_track_id is not None:
                            track_data = tracks.get(det_track_id, {})
                            label_speed = track_data.get("speed_kmh")
                        else:
                            label_speed = None

                        # Цвет по умолчанию — зелёный (объект не превышает лимит
                        # или скорость ещё не рассчитана).
                        box_color = (0, 255, 0)
                        text_color = (0, 255, 0)

                        # Если скорость известна и выше порога — красим в красный.
                        if label_speed is not None and label_speed > SPEED_LIMIT_KMH:
                            box_color = (0, 0, 255)
                            text_color = (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        if label_speed is not None:
                            label = f"person {score:.2f} | {label_speed:.1f} km/h"
                        else:
                            label = f"person {score:.2f}"

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            text_color,
                            2,
                        )

            # Подписи FPS и статуса камеры
            fps_text = f"FPS: {fps_current:.1f}"
            status_text = f"Camera: {camera.index} | Resolution: {frame.shape[1]}x{frame.shape[0]}"

            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, frame)

            # Проверка клавиш выхода (ESC или Q)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q") or key == ord("Q"):  # ESC or Q
                logger.info("Exit requested by user")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error during video stream: {str(e)}", exc_info=True)
        raise
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Video stream ended")
