"""Photographic-subject detection: YOLOv8n running directly on ONNX Runtime.

Ultralytics (and the CUDA-enabled torch stack it drags in) is an export-time
tool only: the runtime ships just the ~12MB .onnx file and onnxruntime, the
same serving path SaliencyEstimator already uses. The pre/post-processing
ultralytics would normally do - letterbox resize, box decoding, NMS - is
implemented here as pure functions so it can be tested against synthetic
tensors without any model download.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from recompose.perception.saliency import default_model_dir
from recompose.types import Box

YOLO_INPUT_SIZE = 640
# Gray value YOLO models are trained to see in letterbox padding.
_PAD_VALUE = 114
# Standard YOLO NMS overlap threshold: boxes of the same class overlapping
# more than this IoU are considered duplicate detections of one object.
NMS_IOU_THRESHOLD = 0.45

# The 80 COCO class names, in training order - index in the output tensor's
# score vector IS the class id, so order here is load-bearing.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

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


def letterbox(
    image: np.ndarray, size: int = YOLO_INPUT_SIZE
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize to fit inside a size x size square without distortion, pad the rest.

    Returns (padded image, scale, (pad_x, pad_y)) - scale and padding are what
    decode_predictions needs to map boxes back to original-image pixels.
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    padded = np.full((size, size, 3), _PAD_VALUE, dtype=image.dtype)
    top, left = int(pad_y), int(pad_x)
    padded[top : top + new_h, left : left + new_w] = resized
    return padded, scale, (pad_x, pad_y)


def decode_predictions(
    output: np.ndarray,
    *,
    scale: float,
    padding: tuple[float, float],
    image_size: tuple[int, int],
    conf: float,
) -> list[Box]:
    """Turn the raw (1, 84, N) YOLOv8 output into subject Boxes in original pixels.

    Each of the N anchor predictions is 4 box values (center x/y, width,
    height, in letterboxed input space) followed by 80 class scores; the max
    class score is the detection confidence (v8 has no objectness score).
    """
    preds = output[0].T  # (N, 84)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    keep = scores >= conf
    subject_mask = np.array(
        [COCO_CLASSES[class_id] in SUBJECT_CLASSES for class_id in class_ids]
    )
    keep &= subject_mask
    if not keep.any():
        return []

    boxes_cxcywh = preds[keep, :4]
    class_ids = class_ids[keep]
    scores = scores[keep]

    # Undo the letterbox: subtract padding, divide by scale, clip to image.
    pad_x, pad_y = padding
    width, height = image_size
    cx, cy, w, h = boxes_cxcywh.T
    x1 = np.clip((cx - w / 2 - pad_x) / scale, 0, width)
    y1 = np.clip((cy - h / 2 - pad_y) / scale, 0, height)
    x2 = np.clip((cx + w / 2 - pad_x) / scale, 0, width)
    y2 = np.clip((cy + h / 2 - pad_y) / scale, 0, height)

    # Per-class NMS: duplicates of one object share a class; a person and the
    # cat they're holding overlap heavily but must both survive.
    result: list[Box] = []
    for class_id in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == class_id)
        xywh = np.stack([x1[idx], y1[idx], x2[idx] - x1[idx], y2[idx] - y1[idx]], axis=1)
        kept = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores[idx].tolist(), conf, NMS_IOU_THRESHOLD
        )
        label = COCO_CLASSES[class_id]
        for i in np.asarray(kept).flatten():
            j = idx[i]
            result.append(
                Box(
                    float(x1[j]),
                    float(y1[j]),
                    float(x2[j]),
                    float(y2[j]),
                    label=label,
                    confidence=float(scores[j]),
                )
            )
    result.sort(key=lambda box: box.confidence, reverse=True)
    return result


def _ensure_weights(model_path: Path) -> Path:
    """Export yolov8n.onnx if missing. Requires ultralytics (dev/build only).

    In the deployed image the .onnx is baked in by a throwaway Docker build
    stage, so this export path never runs in production - and torch never
    enters the runtime image.
    """
    if model_path.exists():
        return model_path
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            f"{model_path} is missing and ultralytics is not installed to "
            "export it. Run `uv sync --all-extras` and retry, or copy a "
            "pre-exported yolov8n.onnx into place."
        ) from exc
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path = model_path.parent / "yolov8n.pt"
    exported = YOLO(str(pt_path)).export(format="onnx", imgsz=YOLO_INPUT_SIZE)
    Path(exported).replace(model_path)
    return model_path


class SubjectDetector:
    """Detects subjects whose boxes act as hard constraints for cropping."""

    def __init__(self, model_path: Path | None = None, conf: float = 0.4):
        import onnxruntime as ort

        path = _ensure_weights(model_path or default_model_dir() / "yolov8n.onnx")
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self.conf = conf

    def detect(self, image: np.ndarray) -> list[Box]:
        """image: HxWx3 RGB uint8. Returns subject boxes in pixel coordinates."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB image, got shape {image.shape}")
        height, width = image.shape[:2]
        padded, scale, padding = letterbox(image)
        # Exported YOLOv8 expects RGB scaled to [0, 1], channels-first.
        blob = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        outputs = self._session.run(None, {self._input_name: blob})
        return decode_predictions(
            outputs[0], scale=scale, padding=padding, image_size=(width, height), conf=self.conf
        )
