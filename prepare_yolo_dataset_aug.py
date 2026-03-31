from __future__ import annotations

from pathlib import Path
import json
import shutil
import random
from typing import Any
from collections import defaultdict, Counter

from PIL import Image
from tqdm import tqdm
import argparse
# ==========================================
# 📍 터미널 실행 인자(Argument) 파싱
# ==========================================
parser = argparse.ArgumentParser(description="PillaTech Dataset Preparation")
parser.add_argument(
    "--exp_name", 
    type=str, 
    default="baseline", 
    help="실험 이름을 입력하세요 (예: baseline, oversampling, copy_paste_v1, copy_paste_v2)"
)
args = parser.parse_args()

EXP_NAME = args.exp_name  # 터미널에서 입력받은 값으로 자동 설정됨!

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / EXP_NAME
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data"
TRAIN_IMG_DIR = RAW_DATA_DIR / "train_images"
AUGMENTED_IMG_DIR = PROCESSED_DIR / "augmented_images"

MERGED_ANNOTATIONS_PATH = PROCESSED_DIR / "merged_annotations.json"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"
TRAIN_SPLIT_PATH = PROCESSED_DIR / "train_split.json"
VAL_SPLIT_PATH = PROCESSED_DIR / "val_split.json"
RARE_CLASSES_PATH = PROCESSED_DIR / "rare_classes.json"

YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_dataset" / EXP_NAME
YOLO_IMAGES_TRAIN = YOLO_ROOT / "images" / "train"
YOLO_IMAGES_VAL = YOLO_ROOT / "images" / "val"
YOLO_LABELS_TRAIN = YOLO_ROOT / "labels" / "train"
YOLO_LABELS_VAL = YOLO_ROOT / "labels" / "val"
YOLO_DATA_YAML = YOLO_ROOT / "dataset.yaml"


# ==========================================
# 📍 Utils & Base YOLO Preparation
# ==========================================
def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dirs() -> None:
    for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL, YOLO_LABELS_TRAIN, YOLO_LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

def xywh_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))

def write_label_file(image_info: dict[str, Any], label_map: dict[str, int], out_path: Path) -> None:
    img_w = image_info["width"]
    img_h = image_info["height"]
    objects = image_info["objects"]
    lines: list[str] = []

    for obj in objects:
        label = obj["label"]
        bbox = obj["bbox"]
        class_id = label_map[label]
        x_center, y_center, w_norm, h_norm = xywh_to_yolo(bbox, img_w, img_h)
        x_center = clip01(x_center)
        y_center = clip01(y_center)
        w_norm = clip01(w_norm)
        h_norm = clip01(h_norm)

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def copy_image(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {src}")
    shutil.copy2(src, dst)

def find_source_image_path(image_name: str) -> Path:
    candidates = [TRAIN_IMG_DIR / image_name, AUGMENTED_IMG_DIR / image_name]
    for p in candidates:
        if p.exists():
            return p
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: '{image_name}'\n확인된 경로:\n{searched}")

def save_dataset_yaml(label_map: dict[str, int]) -> None:
    id_to_label = {idx: label for label, idx in label_map.items()}
    names = [id_to_label[i] for i in range(len(id_to_label))]
    yaml_text = "\n".join([
        f"path: {YOLO_ROOT.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
        *[f"  {i}: '{name}'" for i, name in enumerate(names)],
    ])
    YOLO_DATA_YAML.write_text(yaml_text, encoding="utf-8")


# ==========================================
# 📍 Phase 3: Copy-Paste V2 전용 증강 함수
# ==========================================
def check_overlap(new_box: tuple[float, float, float, float], existing_boxes: list[tuple[float, float, float, float]], margin: int = 10) -> bool:
    for box in existing_boxes:
        if not (new_box[2] + margin < box[0] or new_box[0] > box[2] + margin or 
                new_box[3] + margin < box[1] or new_box[1] > box[3] + margin):
            return True
    return False

def run_copy_paste_augmentation(target_count: int = 50, imgsz: int = 640) -> None:
    if not RARE_CLASSES_PATH.exists():
        print(f"⚠️ {RARE_CLASSES_PATH} 파일이 없습니다.")
        return

    rare_class_names = read_json(RARE_CLASSES_PATH) 
    label_map = read_json(LABEL_MAP_PATH)
    rare_class_ids = [label_map[name] for name in rare_class_names if name in label_map]
    
    print(f"\n🚀 [Phase 3] {len(rare_class_ids)}개의 희귀 클래스 대상 V2(실제 배경 합성) 증강 시작")
    
    crops_db = defaultdict(list)
    background_images = [] # 실제 배경 후보군 저장
    
    label_files = list(YOLO_LABELS_TRAIN.glob("*.txt"))
    for lbl_path in tqdm(label_files, desc="알약 조각 및 배경 추출 중"):
        img_path = YOLO_IMAGES_TRAIN / f"{lbl_path.stem}.png"
        if not img_path.exists(): img_path = YOLO_IMAGES_TRAIN / f"{lbl_path.stem}.jpg"
        if not img_path.exists(): continue
            
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            # 배경 후보로 원본 이미지 저장 (나중에 리사이즈해서 쓸 예정)
            background_images.append(img_path) 
            
            for line in lbl_path.read_text(encoding="utf-8").strip().split('\n'):
                if not line: continue
                parts = line.split()
                cls_id = int(parts[0])
                
                cx, cy, nw, nh = map(float, parts[1:])
                x1, y1 = int((cx - nw/2)*img_w), int((cy - nh/2)*img_h)
                x2, y2 = int((cx + nw/2)*img_w), int((cy + nh/2)*img_h)
                
                # 투명도 처리를 위해 RGBA로 변환하여 추출
                crop = img.crop((max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))).convert("RGBA")
                crops_db[cls_id].append(crop)

    aug_idx = 0
    images_per_rare_class = 15 # 너무 많이 만들면 오히려 독이 될 수 있어 15장으로 조정

    for cls_id in tqdm(rare_class_ids, desc="희귀 알약 실전 합성 중"):
        if not crops_db[cls_id]: continue

        for i in range(images_per_rare_class):
            # 1. 밋밋한 회색 대신 실제 데이터셋 이미지 중 하나를 배경으로 선택
            bg_path = random.choice(background_images)
            with Image.open(bg_path).convert("RGBA") as bg_raw:
                # 훈련용 해상도(640)에 맞게 배경 리사이즈
                bg = bg_raw.resize((imgsz, imgsz))
            
            new_labels, boxes = [], []
            
            # 2. 선택된 배경 위에 희귀 알약을 1~2개만 추가 (기존 배경의 알약과 섞이게)
            for _ in range(random.randint(1, 2)):
                pill = random.choice(crops_db[cls_id]).rotate(random.randint(0, 360), expand=True)
                
                # 크기도 랜덤하게 조절 (0.8배 ~ 1.1배)
                scale = random.uniform(0.8, 1.1)
                pill = pill.resize((int(pill.width * scale), int(pill.height * scale)))

                for _ in range(20): # 배치 시도
                    px, py = random.randint(50, max(50, imgsz-pill.width-50)), random.randint(50, max(50, imgsz-pill.height-50))
                    new_box = (px, py, px + pill.width, py + pill.height)
                    
                    if not check_overlap(new_box, boxes):
                        # 실제 합성 (투명도 마스크 사용)
                        bg.paste(pill, (px, py), pill)
                        boxes.append(new_box)
                        new_labels.append(f"{cls_id} {(px+pill.width/2)/imgsz:.6f} {(py+pill.height/2)/imgsz:.6f} {pill.width/imgsz:.6f} {pill.height/imgsz:.6f}")
                        break
            
            if new_labels:
                save_name = f"aug_real_cp_cls{cls_id}_{i}"
                # 최종 저장은 RGB로 변환 (YOLO 표준)
                bg.convert("RGB").save(YOLO_IMAGES_TRAIN / f"{save_name}.jpg", quality=90)
                (YOLO_LABELS_TRAIN / f"{save_name}.txt").write_text("\n".join(new_labels), encoding="utf-8")
                aug_idx += 1
                
    print(f"✅ 개선된 증강 완료: 총 {aug_idx}장의 실전형 이미지가 추가되었습니다.")

# ==========================================
# 📍 Main
# ==========================================
def main() -> None:
    ensure_dirs()

    missing = [
        p for p in [MERGED_ANNOTATIONS_PATH, LABEL_MAP_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH]
        if not p.exists()
    ]
    if missing:
        missing_list = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing processed artifacts:\n{missing_list}")

    merged = read_json(MERGED_ANNOTATIONS_PATH)
    label_map = read_json(LABEL_MAP_PATH)
    train_images = read_json(TRAIN_SPLIT_PATH)
    val_images = read_json(VAL_SPLIT_PATH)

    images_dict: dict[str, Any] = merged["images"]

    # 1. Train 데이터 복사 (오버샘플링 중복 처리 포함)
    print(f"Train 데이터 준비 중... (총 {len(train_images)}개)")
    train_name_counter = Counter()
    
    for image_name in train_images:
        image_info = images_dict[image_name]
        src_img = find_source_image_path(image_name)
        
        # 중복된 파일명(오버샘플링) 처리
        count = train_name_counter[image_name]
        train_name_counter[image_name] += 1
        
        if count == 0:
            dst_img_name = image_name
            dst_label_name = f"{Path(image_name).stem}.txt"
        else:
            stem = Path(image_name).stem
            ext = Path(image_name).suffix
            dst_img_name = f"{stem}_oversample_{count}{ext}"
            dst_label_name = f"{stem}_oversample_{count}.txt"

        dst_img = YOLO_IMAGES_TRAIN / dst_img_name
        dst_label = YOLO_LABELS_TRAIN / dst_label_name

        copy_image(src_img, dst_img)
        write_label_file(image_info, label_map, dst_label)

    # 2. Val 데이터 복사
    print(f"Val 데이터 복사 중... (총 {len(val_images)}개)")
    for image_name in val_images:
        image_info = images_dict[image_name]
        src_img = find_source_image_path(image_name)
        dst_img = YOLO_IMAGES_VAL / image_name
        dst_label = YOLO_LABELS_VAL / f"{Path(image_name).stem}.txt"

        copy_image(src_img, dst_img)
        write_label_file(image_info, label_map, dst_label)

    # 3. Copy-Paste V2 실험일 때만 추가 증강 실행
    if EXP_NAME == "copy_paste_v2":
        run_copy_paste_augmentation()

    # 4. dataset.yaml 저장
    save_dataset_yaml(label_map)

    print(f"\n✨ YOLO dataset for '{EXP_NAME}' prepared successfully.")
    print(f"📁 dataset.yaml: {YOLO_DATA_YAML}")

if __name__ == "__main__":
    main()