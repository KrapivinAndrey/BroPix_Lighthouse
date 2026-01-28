"""
Object detection module for video frames.

This module provides functionality to detect objects in video frames
using YOLO model and return bounding boxes for visualization.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(
        f"ultralytics not available. Object detection will be disabled. "
        f"Install it with: pip install ultralytics. Error: {str(e)}"
    )


class Detection:
    """
    Represents a single detected object.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        confidence: Confidence score (0.0 to 1.0)
        class_id: Class ID of detected object
        class_name: Name of detected class
        track_id: Optional tracking ID for object tracking across frames
    """

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int, class_name: str, track_id: Optional[int] = None):
        """
        Initialize detection.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            confidence: Confidence score (0.0 to 1.0)
            class_id: Class ID
            class_name: Class name
            track_id: Optional tracking ID for object tracking
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = track_id

    def __repr__(self) -> str:
        """String representation of detection."""
        track_info = f", track_id={self.track_id}" if self.track_id is not None else ""
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox}{track_info})"


class ObjectDetector:
    """
    Object detector using YOLO model.

    Provides methods to detect objects in video frames and return
    bounding boxes with class information.
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize object detector.

        Args:
            model_path: Path to YOLO model file. If None, uses default YOLOv8n model.
            confidence_threshold: Minimum confidence threshold for detections (0.0 to 1.0)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package is required for object detection. Install it with: pip install ultralytics")

        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None

        try:
            if model_path is None:
                # Use default YOLOv8n (nano) model - lightweight for Raspberry Pi
                logger.info("Loading default YOLOv8n model...")
                self.model = YOLO("yolov8n.pt")
            else:
                logger.info(f"Loading YOLO model from {model_path}...")
                self.model = YOLO(model_path)

            logger.info("Object detector initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize object detector: {str(e)}") from e

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a video frame.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            List of Detection objects with bounding boxes and class information
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            detections = []
            if results and len(results) > 0:
                result = results[0]

                # Extract detections
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]

                        detections.append(Detection((x1, y1, x2, y2), confidence, class_id, class_name))

            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}", exc_info=True)
            return []

    def track(self, frame: np.ndarray) -> List[Detection]:
        """
        Track objects in a video frame using YOLO tracking.

        This method uses YOLO's built-in tracking to follow objects across frames.
        It is more efficient than full detection as it tracks already detected objects.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            List of Detection objects with bounding boxes, class information, and track_id
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            # Run tracking inference
            results = self.model.track(frame, conf=self.confidence_threshold, verbose=False, persist=True)

            detections = []
            if results and len(results) > 0:
                result = results[0]

                # Extract tracked detections
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]

                        # Get track_id if available
                        track_id = None
                        if box.id is not None and len(box.id) > 0:
                            track_id = int(box.id[0].cpu().numpy())

                        detections.append(Detection((x1, y1, x2, y2), confidence, class_id, class_name, track_id))

            return detections

        except Exception as e:
            logger.error(f"Error during object tracking: {str(e)}", exc_info=True)
            return []

    def detect_new_objects(self, frame: np.ndarray, existing_track_ids: set) -> List[Detection]:
        """
        Detect only new objects that are not already being tracked.

        This method performs full detection and filters out objects that are already
        being tracked based on their track_id.

        Args:
            frame: Input frame as numpy array (BGR format)
            existing_track_ids: Set of track IDs that are already being tracked

        Returns:
            List of Detection objects representing only new objects (not in existing_track_ids)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            # Run tracking inference to get all objects with track_ids
            results = self.model.track(frame, conf=self.confidence_threshold, verbose=False, persist=True)

            new_detections = []
            if results and len(results) > 0:
                result = results[0]

                # Extract detections
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get track_id
                        track_id = None
                        if box.id is not None and len(box.id) > 0:
                            track_id = int(box.id[0].cpu().numpy())

                        # Only include if this is a new object (not in existing_track_ids)
                        if track_id is None or track_id not in existing_track_ids:
                            # Get bounding box coordinates (x1, y1, x2, y2)
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                            # Get confidence and class
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[class_id]

                            new_detections.append(Detection((x1, y1, x2, y2), confidence, class_id, class_name, track_id))

            return new_detections

        except Exception as e:
            logger.error(f"Error during new object detection: {str(e)}", exc_info=True)
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Detection], new_objects: Optional[List[Detection]] = None, object_speeds: Optional[dict[int, float]] = None, max_speed_kmh: Optional[float] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame for debugging.

        Args:
            frame: Input frame to draw on
            detections: List of Detection objects to draw
            new_objects: Optional list of new Detection objects (will be drawn with different color)
            object_speeds: Optional dictionary mapping track_id to speed in km/h
            max_speed_kmh: Optional maximum speed threshold in km/h for red frame indication

        Returns:
            Frame with drawn bounding boxes and labels
        """
        output_frame = frame.copy()

        # Create set of new object track_ids for quick lookup
        new_track_ids = set()
        if new_objects is not None:
            new_track_ids = {d.track_id for d in new_objects if d.track_id is not None}

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # Get speed for this object if available
            speed_kmh = None
            if object_speeds is not None and detection.track_id is not None:
                speed_kmh = object_speeds.get(detection.track_id)

            # Determine color based on speed threshold and whether this is a new object
            is_new = detection.track_id is not None and detection.track_id in new_track_ids
            is_overspeed = False
            
            if speed_kmh is not None and max_speed_kmh is not None and speed_kmh > max_speed_kmh:
                color = (0, 0, 255)  # Red for overspeed
                is_overspeed = True
            elif is_new:
                color = (0, 255, 255)  # Yellow/Cyan for new objects
            else:
                color = (0, 255, 0)  # Green for tracked objects

            thickness = 2
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)

            # Prepare label text
            track_info = f" ID:{detection.track_id}" if detection.track_id is not None else ""
            new_marker = " [NEW]" if is_new else ""
            
            # Add speed information if available
            speed_text = ""
            if speed_kmh is not None:
                speed_text = f" {speed_kmh:.1f} км/ч"
            
            label = f"{detection.class_name} {detection.confidence:.2f}{track_info}{speed_text}{new_marker}"

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background
            cv2.rectangle(
                output_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                output_frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
            )

        return output_frame
