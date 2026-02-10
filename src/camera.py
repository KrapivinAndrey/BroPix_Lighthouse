"""
Camera module for USB camera capture and video streaming.

This module provides functionality to connect to USB cameras,
capture frames, and display video streams for debugging purposes.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

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

    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None, fps: Optional[int] = None):
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
                raise CameraError(f"Failed to open camera at index {self.index}. Camera may be in use or not connected.")

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
            config = json.load(f)
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
    Display video stream from camera in real-time for debugging.

    Shows FPS counter and camera status. Press ESC or Q to exit.

    Args:
        camera: Camera instance to use. If None, creates new camera from config.
        config: Configuration dictionary. If None, loads from config.json.
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

    try:
        camera.connect()

        window_name = "Camera Debug - Press ESC or Q to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_current = 0.0

        logger.info("Starting video stream. Press ESC or Q to exit.")

        while True:
            ret, frame = camera.read_frame()

            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            # Calculate FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps_current = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Draw FPS and status on frame
            fps_text = f"FPS: {fps_current:.1f}"
            status_text = f"Camera: {camera.index} | Resolution: {frame.shape[1]}x{frame.shape[0]}"

            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

            # Check for exit key (ESC or Q)
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
