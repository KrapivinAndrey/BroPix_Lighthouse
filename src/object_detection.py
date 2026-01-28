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
    """

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int, class_name: str):
        """
        Initialize detection.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            confidence: Confidence score (0.0 to 1.0)
            class_id: Class ID
            class_name: Class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        """String representation of detection."""
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


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

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame for debugging.

        Args:
            frame: Input frame to draw on
            detections: List of Detection objects

        Returns:
            Frame with drawn bounding boxes and labels
        """
        output_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # Draw bounding box
            color = (0, 255, 0)  # Green color for bounding box
            thickness = 2
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)

            # Prepare label text
            label = f"{detection.class_name} {detection.confidence:.2f}"

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
