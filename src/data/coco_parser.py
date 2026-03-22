"""
[Design Intent]
src/data/coco_parser.py

역할: 763개 파편화된 COCO JSON → 이미지 단위 통합 데이터 구조
---
  문제 상황
    - 한 이미지에 약이 3~4개 있는데, 어노테이션은 약 하나당 JSON 파일 1개씩 따로 존재.
    - YOLO 학습은 [이미지 1장 = 라벨 파일 1개 = 해당 이미지의 모든 BBox 포함] 구조가 필요.
  
  해결 방법
    - file_name을 기준 키로 삼아 여러 JSON에 흩어진 BBox들을 하나의 이미지 레코드로 통합.
    - 통합된 결과를 ImageRecord(dataclass) 리스트와 category 매핑 dict로 반환.

  사용법
    from src.data.coco_parser import parse_annotations
    records, cat_map = parse_annotations("./train_annotations")
    print(records[0])  # ImageRecord for first image
"""

import json
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────
# 데이터 구조 정의
# ─────────────────────────────────────────────────

@dataclass
class BBoxAnnotation:
    """COCO 형식 바운딩 박스 하나."""
    category_id:   int
    category_name: str
    bbox:          List[float]   # COCO: [x_min, y_min, width, height] (절대 픽셀)
    area:          Optional[float] = None
    iscrowd:       int = 0

    @property
    def x_min(self):  return self.bbox[0]
    @property
    def y_min(self):  return self.bbox[1]
    @property
    def bw(self):     return self.bbox[2]
    @property
    def bh(self):     return self.bbox[3]


@dataclass
class ImageRecord:
    """이미지 1장에 담긴 모든 정보 (이미지 메타 + 모든 BBox)."""
    image_id:   int
    file_name:  str          # 실제 PNG 파일명
    width:      int
    height:     int
    annotations: List[BBoxAnnotation] = field(default_factory=list)

    # 메타데이터 (EDA용 / 선택)
    drug_shape:   str = ""
    back_color:   str = ""
    drug_S:       str = ""
    color_class1: str = ""


# ─────────────────────────────────────────────────
# 파서 메인 함수
# ─────────────────────────────────────────────────

def parse_annotations(
    ann_dir: str,
    verbose: bool = True
) -> Tuple[List[ImageRecord], Dict[int, str]]:
    """
    train_annotations/ 하위의 모든 JSON을 스캔해
    이미지 파일명 기준으로 통합된 ImageRecord 리스트를 반환한다.

    Args:
        ann_dir : train_annotations 폴더 경로 (str or Path)
        verbose : 진행 상황 출력 여부

    Returns:
        records  : List[ImageRecord] — 이미지 n장에 대한 통합 레코드
        cat_map  : Dict[int, str]   — {category_id: category_name}
    """
    ann_dir = Path(ann_dir)
    json_files = sorted(glob.glob(str(ann_dir / "**" / "*.json"), recursive=True))

    if verbose:
        print(f"[Parser] JSON 파일 수: {len(json_files)}")

    cat_map: Dict[int, str] = {}

    # key: file_name(str) → ImageRecord
    # 동일 이미지에 대한 여러 JSON을 하나의 레코드로 병합
    file_to_record: Dict[str, ImageRecord] = {}
    skipped = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {jf}: {e}")
            skipped += 1
            continue

        # ── categories 수집 ──
        for cat in data.get("categories", []):
            cat_map[cat["id"]] = cat.get("name", f"id_{cat['id']}")

        # ── image 수집 ──
        images_in_file: Dict[int, dict] = {}
        for img in data.get("images", []):
            img_id   = img.get("id")
            filename = img.get("file_name", "")
            if not filename:
                continue

            images_in_file[img_id] = img

            # 이미 등록된 파일명이면 스킵 (중복 이미지 레코드 방지)
            if filename not in file_to_record:
                file_to_record[filename] = ImageRecord(
                    image_id   = img_id,
                    file_name  = filename,
                    width      = img.get("width", 0),
                    height     = img.get("height", 0),
                    drug_shape = img.get("drug_shape", ""),
                    back_color = img.get("back_color", ""),
                    drug_S     = img.get("drug_S", ""),
                    color_class1 = img.get("color_class1", ""),
                )

        # ── annotations 수집 → ImageRecord에 추가 ──
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox", [])
            if not bbox or len(bbox) != 4:
                continue  # 유효하지 않은 BBox 제외

            cat_id  = ann.get("category_id")
            img_id  = ann.get("image_id")

            # image_id → file_name 역추적
            img_meta = images_in_file.get(img_id)
            if img_meta is None:
                continue

            filename = img_meta.get("file_name", "")
            if filename not in file_to_record:
                continue

            # Invalid BBox 필터링 (범위 초과 감지)
            img_w = img_meta.get("width", 0)
            img_h = img_meta.get("height", 0)
            x_min, y_min, bw, bh = bbox
            if img_w > 0 and img_h > 0:
                if x_min >= img_w or y_min >= img_h or x_min < 0 or y_min < 0:
                    if verbose:
                        print(f"  [FILTERED] Invalid BBox (out of bounds) in {filename}: {bbox}")
                    continue

            ann_obj = BBoxAnnotation(
                category_id   = cat_id,
                category_name = cat_map.get(cat_id, f"id_{cat_id}"),
                bbox          = bbox,
                area          = ann.get("area"),
                iscrowd       = ann.get("iscrowd", 0),
            )
            file_to_record[filename].annotations.append(ann_obj)

    # 어노테이션이 0개인 이미지 제거 (안전장치)
    records = [r for r in file_to_record.values() if len(r.annotations) > 0]
    records.sort(key=lambda r: r.file_name)

    if verbose:
        print(f"[Parser] 통합 완료")
        print(f"  고유 이미지    : {len(records)}장")
        print(f"  총 어노테이션  : {sum(len(r.annotations) for r in records)}개")
        print(f"  카테고리 수    : {len(cat_map)}개")
        print(f"  JSON 스킵      : {skipped}개")

    return records, cat_map


# ─────────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ann_dir = sys.argv[1] if len(sys.argv) > 1 else "./train_annotations"
    records, cat_map = parse_annotations(ann_dir, verbose=True)

    print("\n[샘플] 첫 번째 ImageRecord:")
    r = records[0]
    print(f"  file_name : {r.file_name}")
    print(f"  size      : {r.width} × {r.height}")
    print(f"  약 개수   : {len(r.annotations)}")
    for ann in r.annotations:
        print(f"    [{ann.category_id}] {ann.category_name} | bbox={ann.bbox}")
