import argparse
import os
import pandas as pd
import re
import yaml
import json
import difflib 
from ultralytics import YOLO
from pathlib import Path

# --- DEFAULT PATHS --- #
# 현재 파일 위치(/src/data/test_custom.py) 기준으로 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEST_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "test_images"
DEFAULT_YAML_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"
DEFAULT_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "train_annotations"

def parse_args():
    parser = argparse.ArgumentParser(description="Professional Inference Script for Kaggle Submission (Custom Version)")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt model")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output filename")
    parser.add_argument("--data", type=str, default=str(DEFAULT_YAML_PATH), help="Path to dataset.yaml")
    parser.add_argument("--test_images", type=str, default=str(DEFAULT_TEST_IMG_DIR), help="Path to test images folder")
    parser.add_argument("--json_dir", type=str, default=str(DEFAULT_JSON_DIR), help="Path to raw COCO annotations to build label map")
    return parser.parse_args()

def run_test_and_save_csv():
    args = parse_args()
    
    # 인자값에서 설정값 추출
    MODEL_PATH = args.model
    TEST_IMG_DIR = args.test_images
    OUTPUT_CSV = args.output
    YAML_PATH = args.data
    JSON_DIR = args.json_dir

    # ---------------------------------------------------------
    # 2. 스마트 역매핑 딕셔너리 만들기 (difflib 유사도 검사)
    # ---------------------------------------------------------
    # A. YAML 파일 읽기
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        yolo_names = yaml.safe_load(f)['names']
        
    json_category_map = {} 
    original_json_names = {} 

    print(f"🔍 '{JSON_DIR}' 폴더 탐색을 시작합니다.")
    
    for root, dirs, files in os.walk(JSON_DIR):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'categories' in data:
                            for cat in data['categories']:
                                clean_name = str(cat['name']).replace(" ", "")
                                json_category_map[clean_name] = int(cat['id'])
                                original_json_names[clean_name] = cat['name']
                except:
                    pass
    
    print(f"✅ JSON에서 총 {len(json_category_map)}개의 고유 알약 카테고리를 찾아냈습니다.\n")

    # C. 스마트 매핑 시작
    inverse_class_map = {}
    unmapped_classes = []

    for yolo_id, yaml_name in yolo_names.items():
        clean_yaml = str(yaml_name).replace(" ", "")
        
        if clean_yaml in json_category_map:
            inverse_class_map[int(yolo_id)] = json_category_map[clean_yaml]
        else:
            possible_matches = difflib.get_close_matches(clean_yaml, json_category_map.keys(), n=1, cutoff=0.5)
            if possible_matches:
                best_match = possible_matches[0]
                matched_id = json_category_map[best_match]
                inverse_class_map[int(yolo_id)] = matched_id
                print(f"💡 자동 보정 성공: '{yaml_name}' -> JSON의 '{original_json_names[best_match]}' (ID: {matched_id})")
            else:
                unmapped_classes.append(yaml_name)

    if not unmapped_classes:
        print("\n🎉 모든 알약의 클래스 매핑이 완벽하게 완료되었습니다!\n")

    # ---------------------------------------------------------
    # 3. 모델 로드 및 추론
    # ---------------------------------------------------------
    model = YOLO(MODEL_PATH)
    
    if not os.path.exists(TEST_IMG_DIR):
        raise FileNotFoundError(f"Test image directory not found: {TEST_IMG_DIR}")

    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int("".join(re.findall(r'\d+', x))))
    
    results_list = []
    ann_id_counter = 1 

    print(f"🚀 총 {len(image_files)}장의 이미지에 대해 검출을 시작합니다.")
    for idx, img_name in enumerate(image_files):
        if idx % 100 == 0:
            print(f"   - Processing image {idx}/{len(image_files)}...")
            
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        image_id_str = "".join(re.findall(r'\d+', img_name))
        image_id = int(image_id_str) if image_id_str else 0
            
        outputs = model.predict(source=img_path, conf=args.conf, imgsz=args.imgsz, verbose=False)
        
        for r in outputs:
            boxes = r.boxes
            for box in boxes:
                yolo_cls_id = int(box.cls[0]) 
                score = float(box.conf[0])
                real_category_id = inverse_class_map.get(yolo_cls_id, -1)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                results_list.append({
                    'annotation_id': ann_id_counter,
                    'image_id': image_id,
                    'category_id': real_category_id,
                    'bbox_x': int(x1),
                    'bbox_y': int(y1),
                    'bbox_w': int(x2 - x1),
                    'bbox_h': int(y2 - y1),
                    'score': round(score, 3)
                })
                ann_id_counter += 1

    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 분석 완료! 파일 저장됨: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == '__main__':
    run_test_and_save_csv()
