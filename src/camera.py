"""
Camera module for USB camera capture and video streaming.

This module provides functionality to connect to USB cameras,
capture frames, and display video streams for debugging purposes.
"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Exception raised for camera-related errors."""

    pass


class MovementTracker:
    """
    Tracks object movement across frames to filter out static objects.
    
    Maintains a history of object positions and calculates movement
    based on displacement between frames.
    """

    def __init__(self, min_movement_pixels: float = 5.0, history_size: int = 3):
        """
        Initialize movement tracker.

        Args:
            min_movement_pixels: Minimum pixel displacement to consider object as moving
            history_size: Number of previous positions to keep for each object
        """
        self.min_movement_pixels = min_movement_pixels
        self.history_size = history_size
        # Dictionary: track_id -> deque of (center_x, center_y) positions
        self.position_history: dict[int, deque] = {}

    def _calculate_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Calculate center point of bounding box.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2)

        Returns:
            Tuple of (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return center_x, center_y

    def _calculate_displacement(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Distance in pixels
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return np.sqrt(dx * dx + dy * dy)

    def is_moving(self, track_id: Optional[int], bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if object is moving based on position history.

        Args:
            track_id: Tracking ID of the object (None if not tracked)
            bbox: Current bounding box as (x1, y1, x2, y2)

        Returns:
            True if object is moving, False if static
        """
        if track_id is None:
            # New objects without track_id are considered potentially moving
            return True

        current_center = self._calculate_center(bbox)

        if track_id not in self.position_history:
            # First time seeing this object, initialize history
            self.position_history[track_id] = deque(maxlen=self.history_size)
            self.position_history[track_id].append(current_center)
            # Consider new objects as potentially moving
            return True

        # Get previous position
        history = self.position_history[track_id]
        if len(history) == 0:
            history.append(current_center)
            return True

        previous_center = history[-1]
        displacement = self._calculate_displacement(previous_center, current_center)

        # Update history
        history.append(current_center)

        # Object is moving if displacement exceeds threshold
        return displacement >= self.min_movement_pixels

    def remove_track(self, track_id: int) -> None:
        """
        Remove tracking history for an object that is no longer detected.

        Args:
            track_id: Tracking ID to remove
        """
        if track_id in self.position_history:
            del self.position_history[track_id]


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

    Shows FPS counter, camera status, and detected objects with bounding boxes.
    Press ESC or Q to exit.

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

    # Initialize object detector
    detector = None
    try:
        from src.object_detection import ObjectDetector, ULTRALYTICS_AVAILABLE

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Object detection not available. Install ultralytics: pip install ultralytics")
        else:
            detector_config = config.get("object_detection", {}) if config else {}
            confidence_threshold = detector_config.get("confidence_threshold", 0.25)
            detector = ObjectDetector(confidence_threshold=confidence_threshold)
            logger.info("Object detector initialized successfully")
    except ImportError as e:
        logger.warning(f"Object detection not available. Install ultralytics: pip install ultralytics. Error: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to initialize object detector: {str(e)}. Continuing without detection.")

    # Get detection interval from config (default: 5 frames)
    detection_config = config.get("object_detection", {}) if config else {}
    detection_interval = detection_config.get("detection_interval", 5)
    min_movement_pixels = detection_config.get("min_movement_pixels", 5.0)
    logger.info(f"Detection interval set to {detection_interval} frames")
    logger.info(f"Minimum movement threshold: {min_movement_pixels} pixels")

    # Initialize movement tracker
    movement_tracker = MovementTracker(min_movement_pixels=min_movement_pixels)

    try:
        camera.connect()

        window_name = "Camera Debug - Press ESC or Q to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_current = 0.0

        # Object tracking state
        frame_count = 0
        tracked_objects: dict = {}  # track_id -> Detection
        new_detections: list = []  # New objects detected in current frame

        logger.info("Starting video stream. Press ESC or Q to exit.")

        while True:
            ret, frame = camera.read_frame()

            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            frame_count += 1

            # Detect objects if detector is available
            detections = []
            if detector is not None:
                try:
                    if frame_count % detection_interval == 0:
                        # Full detection for new objects
                        all_detections = detector.track(frame)
                        
                        # Save previous track IDs before updating
                        previous_track_ids = set(tracked_objects.keys())
                        
                        # Filter out static objects and update tracked objects dictionary
                        current_track_ids = set()
                        moving_detections = []
                        for det in all_detections:
                            if det.track_id is not None:
                                # Check if object is moving
                                if movement_tracker.is_moving(det.track_id, det.bbox):
                                    moving_detections.append(det)
                                    tracked_objects[det.track_id] = det
                                    current_track_ids.add(det.track_id)
                                else:
                                    # Static object - remove from tracking
                                    movement_tracker.remove_track(det.track_id)
                        
                        # Find new objects (those not in previous tracked_objects)
                        new_detections = [det for det in moving_detections if det.track_id is not None and det.track_id not in previous_track_ids]
                        
                        # Remove objects that are no longer detected or are static
                        tracked_objects = {tid: tracked_objects[tid] for tid in current_track_ids if tid in tracked_objects}
                        
                        # Clean up movement tracker for objects no longer detected
                        for tid in list(movement_tracker.position_history.keys()):
                            if tid not in current_track_ids:
                                movement_tracker.remove_track(tid)
                        
                        detections = moving_detections
                        logger.debug(f"Full detection: {len(all_detections)} total, {len(moving_detections)} moving, {len(new_detections)} new")
                    else:
                        # Track existing objects
                        tracked_detections = detector.track(frame)
                        
                        # Filter out static objects
                        moving_detections = []
                        for det in tracked_detections:
                            if det.track_id is not None:
                                # Check if object is moving
                                if movement_tracker.is_moving(det.track_id, det.bbox):
                                    moving_detections.append(det)
                                    tracked_objects[det.track_id] = det
                                else:
                                    # Static object - remove from tracking
                                    movement_tracker.remove_track(det.track_id)
                        
                        # Update tracked objects and remove static ones
                        current_track_ids = {det.track_id for det in moving_detections if det.track_id is not None}
                        tracked_objects = {tid: tracked_objects[tid] for tid in current_track_ids if tid in tracked_objects}
                        
                        # Clean up movement tracker for objects no longer detected
                        for tid in list(movement_tracker.position_history.keys()):
                            if tid not in current_track_ids:
                                movement_tracker.remove_track(tid)
                        
                        detections = moving_detections
                        new_detections = []  # No new objects when just tracking
                    
                    # Draw bounding boxes on frame
                    frame = detector.draw_detections(frame, detections, new_objects=new_detections)
                except Exception as e:
                    logger.warning(f"Error during object detection: {str(e)}")
                    new_detections = []

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
            objects_text = f"Moving objects: {len(detections)} (Tracked: {len(tracked_objects)}, New: {len(new_detections)})"
            detection_mode_text = f"Mode: {'Detection' if frame_count % detection_interval == 0 else 'Tracking'}"

            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, objects_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, detection_mode_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

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
