from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import random
from typing import Any

SEED = 42
VAL_RATIO = 0.2
RARE_CLASS_THRESHOLD = 5  # 객체 개수가 5개 이하인 클래스를 rare class로 간주
OVERSAMPLE_RARE_IMAGES = True
RARE_IMAGE_DUPLICATION_FACTOR = 2  # rare class 포함 이미지를 총 몇 번 포함할지

random.seed(SEED)

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data"
TRAIN_IMG_DIR = RAW_DATA_DIR / "train_images"
TRAIN_ANN_DIR = RAW_DATA_DIR / "train_annotations"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MERGED_ANNOTATIONS_PATH = PROCESSED_DIR / "merged_annotations.json"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"
DATA_REPORT_PATH = PROCESSED_DIR / "data_report.json"
IMAGE_META_PATH = PROCESSED_DIR / "image_meta.json"
TRAIN_SPLIT_PATH = PROCESSED_DIR / "train_split.json"
VAL_SPLIT_PATH = PROCESSED_DIR / "val_split.json"
RARE_CLASSES_PATH = PROCESSED_DIR / "rare_classes.json"


# =========================
# Utils
# =========================
def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_image_size_from_annotation(
    image_info: dict[str, Any]
) -> tuple[int | None, int | None]:
    width = image_info.get("width")
    height = image_info.get("height")
    return width, height


def validate_bbox_xywh(
    bbox: list[float] | tuple[float, float, float, float],
    width: int | None,
    height: int | None,
) -> tuple[bool, str]:
    """
    bbox format: [x, y, w, h]
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "bbox_format_error"

    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return False, "non_positive_size"

    if x < 0 or y < 0:
        return False, "negative_coordinate"

    if width is not None and x + w > width:
        return False, "x_out_of_bounds"

    if height is not None and y + h > height:
        return False, "y_out_of_bounds"

    return True, "ok"


def bbox_area_ratio(
    bbox: list[float],
    width: int | None,
    height: int | None,
) -> float | None:
    if width is None or height is None or width <= 0 or height <= 0:
        return None

    _, _, w, h = bbox
    return float((w * h) / (width * height))


# =========================
# Step 1. annotation 파일 목록 읽기
# =========================
def load_raw_annotation_files(ann_dir: Path) -> list[Path]:
    return sorted(ann_dir.glob("**/*.json"))


# =========================
# Step 2. 객체 단위 json → 이미지 단위로 병합
# =========================
def merge_annotations_by_image(ann_files: list[Path]) -> dict[str, dict[str, Any]]:
    """
    결과 구조 예시:
    {
      "image1.png": {
        "image_name": "image1.png",
        "width": 976,
        "height": 1280,
        "objects": [
          {
            "bbox": [x, y, w, h],
            "label": "pill_x",
            "category_id": 12,
            "source_json": "xxx.json"
          }
        ]
      }
    }
    """
    merged: dict[str, dict[str, Any]] = {}
    inconsistent_category_names: list[dict[str, Any]] = []

    for ann_file in ann_files:
        ann = read_json(ann_file)

        images = ann.get("images", [])
        categories = ann.get("categories", [])
        annotations = ann.get("annotations", [])

        if not images:
            continue

        image_info = images[0]
        image_name = image_info.get("file_name")
        width, height = get_image_size_from_annotation(image_info)

        if image_name is None:
            continue

        category_map = {cat["id"]: cat["name"] for cat in categories}

        if image_name not in merged:
            merged[image_name] = {
                "image_name": image_name,
                "width": width,
                "height": height,
                "objects": [],
            }
        else:
            prev_w = merged[image_name]["width"]
            prev_h = merged[image_name]["height"]

            if prev_w is None and width is not None:
                merged[image_name]["width"] = width
            if prev_h is None and height is not None:
                merged[image_name]["height"] = height

        for obj in annotations:
            category_id = obj.get("category_id")
            bbox = obj.get("bbox")

            if category_id not in category_map:
                inconsistent_category_names.append({
                    "image_name": image_name,
                    "source_json": ann_file.name,
                    "category_id": category_id,
                    "reason": "category_id_not_found_in_categories",
                })
                label = "UNKNOWN"
            else:
                label = category_map[category_id]

            merged[image_name]["objects"].append({
                "bbox": bbox,
                "label": label,
                "category_id": category_id,
                "source_json": ann_file.name,
            })

    if inconsistent_category_names:
        print(f"[WARN] category mapping issues: {len(inconsistent_category_names)}")

    return merged


# =========================
# Step 3. label map 생성
# =========================
def build_label_map(merged: dict[str, dict[str, Any]]) -> dict[str, int]:
    labels = sorted({
        obj["label"]
        for item in merged.values()
        for obj in item["objects"]
    })
    return {label: idx for idx, label in enumerate(labels)}


# =========================
# Step 4. 데이터 리포트 생성
# =========================
def build_data_report(
    merged: dict[str, dict[str, Any]],
    train_img_dir: Path,
    train_ann_files: list[Path],
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    train_images = sorted(train_img_dir.glob("**/*.*"))
    train_image_names = {p.name for p in train_images}
    merged_image_names = set(merged.keys())

    missing_annotation_images = sorted(list(train_image_names - merged_image_names))
    missing_image_files = sorted(list(merged_image_names - train_image_names))

    class_counter: Counter[str] = Counter()
    objects_per_image_counter: Counter[int] = Counter()
    invalid_bboxes: list[dict[str, Any]] = []
    images_with_invalid_bbox: set[str] = set()
    images_with_zero_objects: list[str] = []
    bbox_area_ratios: list[float] = []

    for image_name, item in merged.items():
        width = item.get("width")
        height = item.get("height")
        objects = item.get("objects", [])

        objects_per_image_counter[len(objects)] += 1

        if len(objects) == 0:
            images_with_zero_objects.append(image_name)

        for obj in objects:
            label = obj["label"]
            bbox = obj["bbox"]

            class_counter[label] += 1

            ok, reason = validate_bbox_xywh(bbox, width, height)
            if not ok:
                invalid_bboxes.append({
                    "image_name": image_name,
                    "label": label,
                    "bbox": bbox,
                    "reason": reason,
                    "source_json": obj.get("source_json"),
                })
                images_with_invalid_bbox.add(image_name)

            ratio = bbox_area_ratio(bbox, width, height)
            if ratio is not None:
                bbox_area_ratios.append(ratio)

    rare_classes = sorted([
        cls_name for cls_name, count in class_counter.items()
        if count <= RARE_CLASS_THRESHOLD
    ])

    report["summary"] = {
        "num_train_images": len(train_images),
        "num_train_annotation_jsons": len(train_ann_files),
        "num_unique_images_in_merged_annotations": len(merged),
        "num_classes": len(class_counter),
        "num_missing_annotation_images": len(missing_annotation_images),
        "num_missing_image_files": len(missing_image_files),
        "num_invalid_bboxes": len(invalid_bboxes),
        "num_images_with_invalid_bbox": len(images_with_invalid_bbox),
        "num_images_with_zero_objects": len(images_with_zero_objects),
    }

    report["class_distribution"] = dict(class_counter)
    report["objects_per_image_distribution"] = dict(sorted(objects_per_image_counter.items()))
    report["rare_classes"] = rare_classes
    report["missing_annotation_images"] = missing_annotation_images
    report["missing_image_files"] = missing_image_files
    report["images_with_zero_objects"] = sorted(images_with_zero_objects)
    report["invalid_bboxes"] = invalid_bboxes[:200]

    if bbox_area_ratios:
        report["bbox_area_ratio_stats"] = {
            "count": len(bbox_area_ratios),
            "min": min(bbox_area_ratios),
            "max": max(bbox_area_ratios),
            "mean": sum(bbox_area_ratios) / len(bbox_area_ratios),
        }
    else:
        report["bbox_area_ratio_stats"] = None

    return report


# =========================
# Step 5. 이미지 단위 메타데이터 생성
# =========================
def build_image_meta(
    merged: dict[str, dict[str, Any]],
    rare_classes: set[str],
) -> list[dict[str, Any]]:
    image_meta: list[dict[str, Any]] = []

    for image_name, item in merged.items():
        objects = item.get("objects", [])
        labels = [obj["label"] for obj in objects]
        label_counter = Counter(labels)

        image_meta.append({
            "image_name": image_name,
            "width": item.get("width"),
            "height": item.get("height"),
            "num_objects": len(objects),
            "labels": sorted(list(set(labels))),
            "label_counts": dict(label_counter),
            "has_rare_class": any(label in rare_classes for label in labels),
        })

    image_meta = sorted(image_meta, key=lambda x: x["image_name"])
    return image_meta


# =========================
# Step 6. train/val 분할
# =========================
def split_train_val_strict(
    image_meta: list[dict[str, Any]],
    merged: dict[str, dict[str, Any]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    목표:
    - validation 비율을 최대한 맞추되
    - 어떤 클래스도 train에서 완전히 사라지지 않도록 분할
    """
    rng = random.Random(seed)

    all_image_names = [x["image_name"] for x in image_meta]
    rng.shuffle(all_image_names)

    target_val_size = max(1, int(len(all_image_names) * val_ratio))

    total_class_counter: Counter[str] = Counter()
    image_to_labels: dict[str, list[str]] = {}

    for image_name in all_image_names:
        objects = merged[image_name]["objects"]
        labels = [obj["label"] for obj in objects]
        image_to_labels[image_name] = labels
        total_class_counter.update(labels)

    train_class_counter = total_class_counter.copy()
    val_images: list[str] = []

    for image_name in all_image_names:
        if len(val_images) >= target_val_size:
            break

        labels_in_image = image_to_labels[image_name]

        can_move_to_val = True
        temp_counter = train_class_counter.copy()

        for label in labels_in_image:
            temp_counter[label] -= 1

        for label in set(labels_in_image):
            if temp_counter[label] <= 0:
                can_move_to_val = False
                break

        if can_move_to_val:
            val_images.append(image_name)
            train_class_counter = temp_counter

    val_images = sorted(val_images)
    val_set = set(val_images)
    train_images = sorted([img for img in all_image_names if img not in val_set])

    return train_images, val_images


def summarize_split(
    split_image_names: list[str],
    merged: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    class_counter: Counter[str] = Counter()
    num_objects_per_image: Counter[int] = Counter()

    for image_name in split_image_names:
        item = merged[image_name]
        objects = item["objects"]

        num_objects_per_image[len(objects)] += 1

        for obj in objects:
            class_counter[obj["label"]] += 1

    return {
        "num_images": len(split_image_names),
        "num_objects": sum(class_counter.values()),
        "num_classes_present": len(class_counter),
        "class_distribution": dict(class_counter),
        "objects_per_image_distribution": dict(sorted(num_objects_per_image.items())),
    }

# =========================
# Step 7. rare class oversampling
# =========================

def oversample_rare_class_images(
    train_images: list[str],
    image_meta: list[dict[str, Any]],
    duplication_factor: int = 2,
) -> list[str]:
    """
    train_images 중 rare class 포함 이미지를 duplication_factor 만큼 반복해서
    oversampled train image list를 반환
    """
    if duplication_factor < 2:
        return sorted(train_images)

    meta_map = {item["image_name"]: item for item in image_meta}

    oversampled_train_images: list[str] = []

    for image_name in train_images:
        oversampled_train_images.append(image_name)

        has_rare_class = meta_map[image_name]["has_rare_class"]
        if has_rare_class:
            for _ in range(duplication_factor - 1):
                oversampled_train_images.append(image_name)

    return sorted(oversampled_train_images)

# =========================
# Main
# =========================
def main() -> None:
    print("=" * 60)
    print("Start preprocessing")
    print("=" * 60)
    print(f"PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"TRAIN_IMG_DIR    : {TRAIN_IMG_DIR}")
    print(f"TRAIN_ANN_DIR    : {TRAIN_ANN_DIR}")
    print(f"PROCESSED_DIR    : {PROCESSED_DIR}")

    if not TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"Train image directory not found: {TRAIN_IMG_DIR}")

    if not TRAIN_ANN_DIR.exists():
        raise FileNotFoundError(f"Train annotation directory not found: {TRAIN_ANN_DIR}")

    ann_files = load_raw_annotation_files(TRAIN_ANN_DIR)
    print(f"[1] Loaded annotation json files: {len(ann_files)}")

    merged = merge_annotations_by_image(ann_files)
    print(f"[2] Merged into image-level annotations: {len(merged)} images")

    label_map = build_label_map(merged)
    print(f"[3] Built label map: {len(label_map)} classes")

    report = build_data_report(
        merged=merged,
        train_img_dir=TRAIN_IMG_DIR,
        train_ann_files=ann_files,
    )

    rare_classes = set(report["rare_classes"])
    print(f"[4] Built report")
    print(f"    - num classes              : {report['summary']['num_classes']}")
    print(f"    - missing annotation imgs  : {report['summary']['num_missing_annotation_images']}")
    print(f"    - missing image files      : {report['summary']['num_missing_image_files']}")
    print(f"    - invalid bboxes           : {report['summary']['num_invalid_bboxes']}")
    print(f"    - rare classes             : {len(rare_classes)}")

    image_meta = build_image_meta(merged, rare_classes)
    print(f"[5] Built image metadata: {len(image_meta)} items")

    train_images, val_images = split_train_val_strict(
    image_meta=image_meta,
    merged=merged,
    val_ratio=VAL_RATIO,
    seed=SEED,

    )
    print(f"[6] Split train/val")
    print(f"    - train images: {len(train_images)}")
    print(f"    - val images  : {len(val_images)}")

    if OVERSAMPLE_RARE_IMAGES:
        train_images = oversample_rare_class_images(
            train_images=train_images,
            image_meta=image_meta,
            duplication_factor=RARE_IMAGE_DUPLICATION_FACTOR,
        )
        print(f"[7] Applied rare class image oversampling")
        print(f"    - train images (after oversampling): {len(train_images)}")

    else:
        print(f"[7] Rare class image oversampling skipped")

    train_summary = summarize_split(train_images, merged)
    val_summary = summarize_split(val_images, merged)

    split_info = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "train_summary": train_summary,
        "val_summary": val_summary,
    }

    merged_output = {
        "metadata": {
            "description": "Merged image-level annotations generated from object-level json files",
            "num_images": len(merged),
            "num_classes": len(label_map),
            "label_map_path": str(LABEL_MAP_PATH.name),
        },
        "images": merged,
    }

    write_json(MERGED_ANNOTATIONS_PATH, merged_output)
    write_json(LABEL_MAP_PATH, label_map)
    write_json(IMAGE_META_PATH, image_meta)
    write_json(TRAIN_SPLIT_PATH, train_images)
    write_json(VAL_SPLIT_PATH, val_images)
    write_json(RARE_CLASSES_PATH, sorted(list(rare_classes)))

    report_with_split = report.copy()
    report_with_split["split_info"] = split_info
    write_json(DATA_REPORT_PATH, report_with_split)

    print("[7] Saved files")
    print(f"    - {MERGED_ANNOTATIONS_PATH}")
    print(f"    - {LABEL_MAP_PATH}")
    print(f"    - {DATA_REPORT_PATH}")
    print(f"    - {IMAGE_META_PATH}")
    print(f"    - {TRAIN_SPLIT_PATH}")
    print(f"    - {VAL_SPLIT_PATH}")
    print(f"    - {RARE_CLASSES_PATH}")

    print("=" * 60)
    print("Preprocessing done.")
    print("=" * 60)


if __name__ == "__main__":
    main()