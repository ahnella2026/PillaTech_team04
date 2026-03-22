"""
[Design Intent]
src/data/preprocessing.py

역할: ImageRecord 리스트(COCO) → YOLO 학습 데이터셋 변환 + Train/Val Stratified Split
---
  변환 공식 (COCO → YOLO)
    COCO bbox: [x_min, y_min, width, height]  (절대 픽셀)
    YOLO label: <class_idx> <x_center> <y_center> <w_norm> <h_norm>  (0~1 정규화)

    x_center = (x_min + width  / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    w_norm   = width  / img_width
    h_norm   = height / img_height

  출력 파일 구조
    data/yolo/
    ├── images/
    │   ├── train/  ← 학습 이미지 (심볼릭 링크)
    │   └── val/    ← 검증 이미지 (심볼릭 링크)
    ├── labels/
    │   ├── train/  ← 학습 라벨 txt
    │   └── val/    ← 검증 라벨 txt
    ├── classes.txt ← 클래스 인덱스 → 이름 목록
    └── dataset.yaml← YOLO 학습 설정 파일

  사용법
    python src/data/coco_to_yolo.py \\
        --ann_dir    ./train_annotations \\
        --img_dir    ./train_images \\
        --output_dir ./data/yolo \\
        --val_ratio  0.2
"""

import argparse
import os
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from dotenv import load_dotenv

# .env 로드 (실무 표준 하드코딩 방지)
load_dotenv()

# 같은 패키지의 파서 import (직접 실행 시 경로 처리)
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.coco_parser import parse_annotations, ImageRecord
from src.data.label_audit import (
    find_missing_labels,
    find_invalid_bboxes,
    bbox_invalid_reason,
    clip_coco_bbox_to_image,
)


# ─────────────────────────────────────────────────
# 좌표 변환 함수
# ─────────────────────────────────────────────────

def coco_to_yolo_bbox(
    bbox: list,
    img_w: int,
    img_h: int
) -> tuple:
    """
    COCO [x_min, y_min, w, h] (절대 px) → YOLO [x_c, y_c, w, h] (0~1 정규화)
    이 변환이 틀리면 모델이 허공을 학습한다.
    """
    x_min, y_min, bw, bh = bbox

    x_center = (x_min + bw / 2) / img_w
    y_center = (y_min + bh / 2) / img_h
    w_norm   = bw / img_w
    h_norm   = bh / img_h

    return x_center, y_center, w_norm, h_norm


# ─────────────────────────────────────────────────
# Stratified Split
# ─────────────────────────────────────────────────

def stratified_split(
    records: list,
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple:
    """
    클래스 비율을 유지하며 Train/Val을 나눈다 (Stratified Split).
    소수 클래스가 val에만 쏠리는 현상을 방지.

    Args:
        records   : List[ImageRecord]
        val_ratio : 검증 비율 (0~1)
        seed      : 재현성을 위한 랜덤 시드

    Returns:
        train_records, val_records
    """
    random.seed(seed)

    # 이미지별 '대표 클래스' = 해당 이미지에서 가장 희귀한 클래스
    # → 소수 클래스 이미지가 최대한 양쪽에 골고루 들어가게
    class_counts = Counter()
    for r in records:
        for ann in r.annotations:
            class_counts[ann.category_id] += 1

    def rarest_class(record: ImageRecord) -> int:
        return min(
            (ann.category_id for ann in record.annotations),
            key=lambda cid: class_counts[cid]
        )

    # 대표 클래스별로 이미지를 그룹핑
    class_to_records = defaultdict(list)
    for r in records:
        class_to_records[rarest_class(r)].append(r)

    train_records, val_records = [], []
    for cls_id, cls_records in class_to_records.items():
        random.shuffle(cls_records)
        n_val = max(1, round(len(cls_records) * val_ratio))
        val_records   += cls_records[:n_val]
        train_records += cls_records[n_val:]

    return train_records, val_records


# ─────────────────────────────────────────────────
# 메인 변환 함수
# ─────────────────────────────────────────────────

def convert_to_yolo(
    records:     list,
    cat_map:     dict,
    img_src_dir: Path,
    output_dir:  Path,
    val_ratio:   float = 0.2,
    seed:        int   = 42,
    copy_images: bool  = False,
    bbox_policy: str   = "skip",
    verbose:     bool  = True
) -> dict:
    """
    ImageRecord 리스트를 YOLO 데이터셋 폴더로 변환한다.

    Args:
        records     : parse_annotations()의 반환값
        cat_map     : {category_id: category_name}
        img_src_dir : 원본 train_images 폴더
        output_dir  : 출력 폴더 (data/yolo/)
        val_ratio   : 검증 비율
        seed        : 스플릿 랜덤 시드
        copy_images : True → 이미지 복사, False → 심볼릭 링크 (디스크 절약)
        verbose     : 진행 출력

    Returns:
        info dict (train/val 개수, 클래스 수 등)
    """
    output_dir = Path(output_dir)
    img_src_dir = Path(img_src_dir)

    # ── 출력 디렉토리 생성 ──
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 클래스 인덱스 매핑 생성 ──
    # category_id를 정렬 후 0부터 순서대로 인덱스 부여
    sorted_cat_ids = sorted(cat_map.keys())
    cat_id_to_idx  = {cid: idx for idx, cid in enumerate(sorted_cat_ids)}

    # ── 라벨 QA (누락/이상치 분리) ──
    missing_label_cases = find_missing_labels(records)
    invalid_bbox_cases = find_invalid_bboxes(records)

    if verbose:
        print("\n[QA] 라벨 누락/이상치 점검")
        print(f"  라벨 누락 이미지 : {len(missing_label_cases)}장")
        print(f"  Invalid BBox     : {len(invalid_bbox_cases)}건 (이미지 {len({c.file_name for c in invalid_bbox_cases})}장)")
        if len(missing_label_cases) > 0:
            print("  - 예시(최대 3개):")
            for c in missing_label_cases[:3]:
                print(f"    {c.file_name} (expected={c.expected}, observed={c.observed})")
        if len(invalid_bbox_cases) > 0:
            print("  - 예시(최대 1개):")
            c = invalid_bbox_cases[0]
            print(f"    {c.file_name} bbox={c.bbox} reason={c.reason}")

    if verbose:
        print(f"\n[Converter] 클래스 인덱스 매핑 ({len(sorted_cat_ids)}개):")
        for cid in sorted_cat_ids[:10]:
            print(f"  idx={cat_id_to_idx[cid]:3d} | id={cid:6d} | {cat_map[cid][:40]}")
        if len(sorted_cat_ids) > 10:
            print(f"  ... (총 {len(sorted_cat_ids)}개)")

    # ── Stratified Split ──
    train_records, val_records = stratified_split(records, val_ratio, seed)

    if verbose:
        print(f"\n[Split] Train: {len(train_records)}장 | Val: {len(val_records)}장")

    # ── 변환 실행 ──
    stats = {
        "train": 0,
        "val": 0,
        "missing_img": 0,
        "invalid_bbox": 0,
        "invalid_category": 0,
    }

    for split, split_records in [("train", train_records), ("val", val_records)]:
        for record in split_records:
            src_img = img_src_dir / record.file_name
            dst_img = output_dir / "images" / split / record.file_name
            dst_lbl = output_dir / "labels"  / split / (Path(record.file_name).stem + ".txt")

            # 이미지 처리 (복사 or 심볼릭 링크)
            if src_img.exists():
                if copy_images:
                    shutil.copy2(src_img, dst_img)
                else:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    dst_img.symlink_to(src_img.resolve())
            else:
                if verbose:
                    print(f"  [WARN] 이미지 없음: {src_img}")
                stats["missing_img"] += 1

            # 라벨 txt 작성
            lines = []
            for ann in record.annotations:
                if ann.iscrowd:
                    continue  # crowd annotation 제외

                class_idx = cat_id_to_idx.get(ann.category_id)
                if class_idx is None:
                    stats["invalid_category"] += 1
                    continue

                bbox = ann.bbox
                reason = bbox_invalid_reason(bbox, record.width, record.height)
                if reason is not None:
                    if bbox_policy == "keep":
                        pass
                    elif bbox_policy == "clip":
                        clipped = clip_coco_bbox_to_image(bbox, record.width, record.height)
                        if clipped is None:
                            stats["invalid_bbox"] += 1
                            continue
                        bbox = clipped
                    elif bbox_policy == "skip":
                        stats["invalid_bbox"] += 1
                        continue
                    else:
                        raise ValueError(f"Unknown bbox_policy: {bbox_policy}")

                xc, yc, wn, hn = coco_to_yolo_bbox(bbox, record.width, record.height)

                # YOLO 라벨은 보통 0~1 범위를 기대한다.
                # bbox_policy=keep로 인해 값이 벗어날 수 있으므로, 필요 시 여기서 정책적으로 차단할 것.

                lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))

            stats[split] += 1

    # ── classes.txt 저장 ──
    classes_path = output_dir / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        for cid in sorted_cat_ids:
            f.write(f"{cat_map[cid]}\n")
    if verbose:
        print(f"[Save] classes.txt → {classes_path} ({len(sorted_cat_ids)}클래스)")

    # ── dataset.yaml 저장 (YOLO 학습 설정) ──
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""# [Design Intent] Ultralytics YOLO 학습용 데이터셋 설정 파일
# 자동 생성됨 — coco_to_yolo.py

path: {output_dir.resolve()}   # 데이터셋 루트
train: images/train
val:   images/val

nc: {len(sorted_cat_ids)}  # 클래스 수

names:
"""
    for cid in sorted_cat_ids:
        yaml_content += f"  - '{cat_map[cid]}'\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    if verbose:
        print(f"[Save] dataset.yaml → {yaml_path}")

    # ── QA 리포트 저장 ──
    audit_path = output_dir / "audit.txt"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("[Label QA]\n")
        f.write(f"- bbox_policy: {bbox_policy}\n")
        f.write(f"- missing_label_images: {len(missing_label_cases)}\n")
        f.write(f"- invalid_bbox_annotations: {len(invalid_bbox_cases)}\n\n")
        if missing_label_cases:
            f.write("[Missing label images]\n")
            for c in missing_label_cases:
                f.write(f"- {c.file_name} (expected={c.expected}, observed={c.observed})\n")
            f.write("\n")
        if invalid_bbox_cases:
            f.write("[Invalid bbox annotations]\n")
            for c in invalid_bbox_cases:
                f.write(f"- {c.file_name} bbox={c.bbox} reason={c.reason}\n")

    if verbose:
        print(f"[Save] audit.txt → {audit_path}")

    # ── 변환 완료 요약 ──
    if verbose:
        print(f"\n{'='*50}")
        print(f"변환 완료!")
        print(f"  Train 라벨 생성  : {stats['train']}개")
        print(f"  Val   라벨 생성  : {stats['val']}개")
        print(f"  이미지 없음      : {stats['missing_img']}개")
        print(f"  Invalid BBox 처리: {stats['invalid_bbox']}건 (policy={bbox_policy})")
        print(f"  Invalid Category : {stats['invalid_category']}건")
        print(f"  출력 폴더        : {output_dir}")
        print(f"{'='*50}")

    return {
        "n_train": stats["train"],
        "n_val":   stats["val"],
        "n_classes": len(sorted_cat_ids),
        "cat_id_to_idx": cat_id_to_idx,
        "output_dir": str(output_dir),
        "bbox_policy": bbox_policy,
        "missing_label_images": len(missing_label_cases),
        "invalid_bbox_annotations": len(invalid_bbox_cases),
    }


# ─────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────

def parse_args():
    # .env 환경 변수에서 기본값 로드 (보험용 디폴트 포함)
    env_seeds   = os.getenv("RANDOM_SEEDS", "42,123,777")
    env_img_dir = os.getenv("DATA_IMG_DIR", "./train_images")
    env_ann_dir = os.getenv("DATA_ANN_DIR", "./train_annotations")
    env_out_dir = os.getenv("DATA_OUTPUT_DIR", "./data/yolo")
    env_val_rat = float(os.getenv("DATA_VAL_RATIO", "0.2"))

    p = argparse.ArgumentParser(description="COCO JSON → YOLO 데이터셋 변환")
    p.add_argument("--ann_dir",    default=env_ann_dir, help="어노테이션 폴더")
    p.add_argument("--img_dir",    default=env_img_dir, help="원본 이미지 폴더")
    p.add_argument("--output_dir", default=env_out_dir, help="출력 폴더")
    p.add_argument("--val_ratio",  type=float, default=env_val_rat, help="검증 비율 (기본: 0.2)")
    p.add_argument("--seeds",      type=str,   default=env_seeds, help="랜덤 시드 (콤마 단위로 여러 개 입력 시 다중 Split 진행)")
    p.add_argument("--copy",       action="store_true",           help="이미지 심볼릭 링크 대신 복사")
    p.add_argument(
        "--bbox_policy",
        choices=["skip", "clip", "keep"],
        default="skip",
        help="Invalid BBox 처리 정책 (skip=제외, clip=경계로 클리핑, keep=원본 유지)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 50)
    print("STEP 1. COCO JSON 파싱")
    print("=" * 50)
    records, cat_map = parse_annotations(args.ann_dir, verbose=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"\n설정된 랜덤 시드: {seeds} (총 {len(seeds)}개 Split 생성)")

    for seed in seeds:
        print("\n" + "=" * 50)
        print(f"STEP 2. YOLO 포맷 변환 + Train/Val Split (Seed: {seed})")
        print("=" * 50)
        
        # 다중 시드일 경우 output_dir 하위에 seed별 폴더 생성
        current_output_dir = Path(args.output_dir) if len(seeds) == 1 else Path(args.output_dir) / f"seed_{seed}"
        
        info = convert_to_yolo(
            records     = records,
            cat_map     = cat_map,
            img_src_dir = Path(args.img_dir),
            output_dir  = current_output_dir,
            val_ratio   = args.val_ratio,
            seed        = seed,
            copy_images = args.copy,
            bbox_policy = args.bbox_policy,
            verbose     = True,
        )

        print(f"\n[완료] Seed {seed} dataset.yaml 경로: {current_output_dir / 'dataset.yaml'}")
        print(f"  yolo train model=yolov8n.pt data={current_output_dir / 'dataset.yaml'}")
