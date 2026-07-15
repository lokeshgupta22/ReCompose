"""Photographic-subject detection using YOLOv8."""

from __future__ import annotations

import numpy as np

from recompose.types import Box

# COCO classes that typically constitute the subject of a photograph.
SUBJECT_CLASSES = {
    "person",
    "cat",
    "dog",
    "bird",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "bicycle",
    "motorcycle",
    "car",
    "boat",
    "airplane",
}


class SubjectDetector:
    """Detects subjects whose boxes act as hard constraints for cropping."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.4):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, image: np.ndarray) -> list[Box]:
        """image: HxWx3 RGB uint8. Returns subject boxes in pixel coordinates."""
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.model.predict(bgr, conf=self.conf, verbose=False)
        boxes: list[Box] = []
        for result in results:
            names = result.names
            for xyxy, cls, conf in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
                strict=True,
            ):
                label = names[int(cls)]
                if label not in SUBJECT_CLASSES:
                    continue
                x1, y1, x2, y2 = xyxy
                boxes.append(Box(x1, y1, x2, y2, label=label, confidence=float(conf)))
        return boxes
