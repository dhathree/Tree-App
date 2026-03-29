from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency runtime fallback
    YOLO = None


@dataclass
class Detection:
    cls: str
    confidence: float
    bbox_xyxy: List[int]


def run_tree_detection(image_bgr: np.ndarray, model_name: str = "yolov8n.pt") -> tuple[np.ndarray, List[Detection], str]:
    """
    Runs YOLO inference if available; otherwise uses fallback vegetation segmentation
    to keep the demo runnable in restricted environments.
    """
    if YOLO is None:
        return _fallback_detection(image_bgr)

    try:
        model = YOLO(model_name)
        results = model.predict(image_bgr, verbose=False)
        plotted = results[0].plot()
        detections: List[Detection] = []

        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and boxes.cls is not None:
            names = results[0].names
            for i in range(len(boxes.cls)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                detections.append(Detection(cls=names.get(cls_id, str(cls_id)), confidence=conf, bbox_xyxy=xyxy))

        return plotted, detections, "YOLOv8"
    except Exception:
        return _fallback_detection(image_bgr)


def _fallback_detection(image_bgr: np.ndarray) -> tuple[np.ndarray, List[Detection], str]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 30, 20], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image_bgr.copy()
    detections: List[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 450:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, "tree_cluster", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detections.append(Detection(cls="tree_cluster", confidence=0.4, bbox_xyxy=[x, y, x + w, y + h]))

    return output, detections, "Fallback CV"
