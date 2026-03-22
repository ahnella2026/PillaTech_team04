"""
src/data/label_audit.py

Dataset label QA utilities:
- Missing label detection (expected object count from image filename vs observed bboxes)
- Invalid bbox detection (out of bounds / non-positive sizes)
- Optional bbox clipping helper (for safe YOLO export)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from src.data.coco_parser import ImageRecord, BBoxAnnotation


def expected_object_count_from_filename(file_name: str) -> Optional[int]:
    """
    Train image naming convention:
      K-<id>-<id>-<id>[_...]  => expected 3 objects
      K-<id>-<id>-<id>-<id>[_...]  => expected 4 objects
    Returns None if pattern is not recognized.
    """
    base = file_name.split("_", 1)[0]
    parts = base.split("-")
    if not parts or parts[0] != "K":
        return None
    n_ids = len(parts) - 1
    if n_ids not in (3, 4):
        return None
    return n_ids


@dataclass(frozen=True)
class MissingLabelCase:
    file_name: str
    expected: int
    observed: int


@dataclass(frozen=True)
class InvalidBBoxCase:
    file_name: str
    bbox: list[float]
    reason: str


def bbox_invalid_reason(bbox: list[float], img_w: int, img_h: int) -> Optional[str]:
    x_min, y_min, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return "non_positive_size"
    if x_min < 0 or y_min < 0:
        return "negative_origin"
    if x_min >= img_w or y_min >= img_h:
        return "origin_out_of_bounds"
    if x_min + bw > img_w or y_min + bh > img_h:
        return "extent_out_of_bounds"
    return None


def clip_coco_bbox_to_image(bbox: list[float], img_w: int, img_h: int) -> Optional[list[float]]:
    """
    Clips COCO bbox [x_min, y_min, bw, bh] into image bounds.
    Returns None if clipping results in a non-positive box.
    """
    x_min, y_min, bw, bh = bbox
    x1 = max(0.0, float(x_min))
    y1 = max(0.0, float(y_min))
    x2 = min(float(img_w), float(x_min + bw))
    y2 = min(float(img_h), float(y_min + bh))
    new_bw = x2 - x1
    new_bh = y2 - y1
    if new_bw <= 0 or new_bh <= 0:
        return None
    return [x1, y1, new_bw, new_bh]


def find_missing_labels(records: Iterable[ImageRecord]) -> list[MissingLabelCase]:
    cases: list[MissingLabelCase] = []
    for r in records:
        expected = expected_object_count_from_filename(r.file_name)
        if expected is None:
            continue
        observed = len(r.annotations)
        if observed < expected:
            cases.append(MissingLabelCase(r.file_name, expected, observed))
    cases.sort(key=lambda c: c.file_name)
    return cases


def find_invalid_bboxes(records: Iterable[ImageRecord]) -> list[InvalidBBoxCase]:
    cases: list[InvalidBBoxCase] = []
    for r in records:
        for ann in r.annotations:
            reason = bbox_invalid_reason(ann.bbox, r.width, r.height)
            if reason is not None:
                cases.append(InvalidBBoxCase(r.file_name, ann.bbox, reason))
    cases.sort(key=lambda c: c.file_name)
    return cases


def audit_summary(records: Iterable[ImageRecord]) -> dict:
    missing = find_missing_labels(records)
    invalid = find_invalid_bboxes(records)
    return {
        "missing_label_images": len(missing),
        "invalid_bbox_annotations": len(invalid),
        "invalid_bbox_images": len({c.file_name for c in invalid}),
    }

