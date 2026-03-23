import os
import json
import shutil
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

# ==========================================
# ⚙️ 기본 설정 (경로 세팅)
# ==========================================
BASE_DIR = 'data/sprint_ai_project1_data'

ANNOTATION_DIR = os.path.join(BASE_DIR, 'train_annotations') 
IMG_DIR = os.path.join(BASE_DIR, 'train_images')             
NEW_DATA_DIR = './pill_dataset_baseline_aug'

if not os.path.exists(ANNOTATION_DIR):
    raise FileNotFoundError(f"🚨 라벨 폴더를 찾을 수 없습니다: {ANNOTATION_DIR}")
if not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"🚨 이미지 폴더를 찾을 수 없습니다: {IMG_DIR}")

# ==========================================
# 📍 Phase 1: 폴더 내 모든 JSON 파싱 및 YOLO 라벨 생성
# ==========================================
def phase1_parse_json():
    print("\n[Phase 1] JSON 파싱 및 YOLO 라벨 변환 (Class ID 재배치 포함)...")
    temp_label_dir = os.path.join(NEW_DATA_DIR, 'temp_labels_all')
    os.makedirs(temp_label_dir, exist_ok=True)
    
    json_files = []
    for root, dirs, files in os.walk(ANNOTATION_DIR):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
                
    raw_categories = {}
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for cat in data.get('categories', []):
                raw_categories[cat['id']] = cat['name']
                
    sorted_orig_ids = sorted(list(raw_categories.keys()))
    class_map = {orig_id: yolo_id for yolo_id, orig_id in enumerate(sorted_orig_ids)}
    yolo_names = [raw_categories[orig_id] for orig_id in sorted_orig_ids]
    
    yaml_data = {
        'path': os.path.abspath(NEW_DATA_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(yolo_names),
        'names': yolo_names
    }
    with open(os.path.join(NEW_DATA_DIR, 'data_basemodel_aug.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True)
    
    for json_path in tqdm(json_files, desc="YOLO 라벨 생성 중"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            img_info = {img['id']: img for img in data.get('images', [])}
            
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in img_info: continue
                
                orig_cls_id = ann['category_id']
                yolo_cls_id = class_map[orig_cls_id] 
                
                img_w = img_info[img_id]['width']
                img_h = img_info[img_id]['height']
                file_name = img_info[img_id]['file_name']
                
                x_min, y_min, w, h = ann['bbox']
                cx = (x_min + w / 2) / img_w
                cy = (y_min + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                txt_filename = os.path.splitext(file_name)[0] + '.txt'
                txt_path = os.path.join(temp_label_dir, txt_filename)
                
                with open(txt_path, 'a', encoding='utf-8') as txt_f:
                    txt_f.write(f"{yolo_cls_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
    print(f"✅ Phase 1 완료! (data.yaml 생성 완료, 총 {len(yolo_names)}개 클래스)")
    return temp_label_dir

# ==========================================
# 📍 Phase 2: 다중 라벨 계층적 분할 (Stratified Split 8:2) + 희귀 데이터 예외 처리
# ==========================================
def phase2_split_data(temp_label_dir):
    print("\n[Phase 2] Train(8) / Val(2) 계층적 분할(Stratified Split) 적용 시작...")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(NEW_DATA_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(NEW_DATA_DIR, split, 'labels'), exist_ok=True)
        
    all_labels = [f for f in os.listdir(temp_label_dir) if f.endswith('.txt')]
    
    stratify_classes = []
    for label_file in all_labels:
        with open(os.path.join(temp_label_dir, label_file), 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = int(first_line.split()[0])
                stratify_classes.append(class_id)
            else:
                stratify_classes.append(-1)
                
    class_counts = Counter(stratify_classes)
    
    abundant_files, abundant_classes, rare_files = [], [], []
    
    for label_file, cls_id in zip(all_labels, stratify_classes):
        if class_counts[cls_id] >= 2:
            abundant_files.append(label_file)
            abundant_classes.append(cls_id)
        else:
            rare_files.append(label_file)
            
    if abundant_files:
        train_abundant, val_labels = train_test_split(
            abundant_files, test_size=0.2, random_state=42, stratify=abundant_classes
        )
    else:
        train_abundant, val_labels = [], []
        
    train_labels = train_abundant + rare_files
    train_labels_set = set(train_labels) 
    
    for label_file in tqdm(all_labels, desc="계층적 데이터 복사 중"):
        base_name = os.path.splitext(label_file)[0]
        img_src = os.path.join(IMG_DIR, base_name + '.png')
        if not os.path.exists(img_src):
            img_src = os.path.join(IMG_DIR, base_name + '.jpg')
            if not os.path.exists(img_src): continue
            
        lbl_src = os.path.join(temp_label_dir, label_file)
        
        target_folder = 'train' if label_file in train_labels_set else 'val'
        img_dst = os.path.join(NEW_DATA_DIR, target_folder, 'images', os.path.basename(img_src))
        lbl_dst = os.path.join(NEW_DATA_DIR, target_folder, 'labels', label_file)
        
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)
        
    print(f"✅ Phase 2 완료! (Train: {len(train_labels)}장, Val: {len(val_labels)}장)")

# ==========================================
# 📍 Phase 4: 베이스라인 모델 학습 (가장 순수한 세팅)
# ==========================================
def phase4_train_model():
    print("\n[Phase 4] 베이스라인 모델 학습 시작 (YOLOv8n / imgsz=640 / 무증강)")
    
    model = YOLO('yolov8n.pt') 
    yaml_path = os.path.join(NEW_DATA_DIR, 'data_basemodel_aug.yaml')
    
    results = model.train(
        data=yaml_path,
        epochs=50,             
        imgsz=640,             
        batch=16,              
        device='mps',          
        mosaic=0.5,         # 🚨 완전히 기본 상태로 만들기 위해 모자이크 증강 끄기
        mixup=1.0,          # 🚨 믹스업 증강 끄기
        project='PillaTech',
        name='exp_baseline' 
    )
    print("\n🎉 전처리 및 순수 베이스라인 학습 파이프라인 전체 완료!")

if __name__ == '__main__':
    temp_labels = phase1_parse_json()
    phase2_split_data(temp_labels)
    phase4_train_model()