import os
import cv2
import pandas as pd
import re
import yaml
import json
import difflib 
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def run_test_and_save_csv():
    # 1. 경로 설정
    MODEL_PATH = '/Users/yewon/Desktop/코드잇 스프린트/project/project1/runs/exp8_yolo11s_clahe3/weights/best.pt'
    TEST_IMG_DIR = '/Users/yewon/Desktop/코드잇 스프린트/project/project1/data/raw/sprint_ai_project1_data/test_images' 
    OUTPUT_CSV = 'Yewon_v5_yolov11s_CLAHE.csv'
    YAML_PATH = '/Users/yewon/Desktop/코드잇 스프린트/project/project1/data/yolo_dataset/dataset.yaml'
    JSON_DIR = '/Users/yewon/Desktop/코드잇 스프린트/project/project1/data/raw/sprint_ai_project1_data/train_annotations' 

    # CLAHE 객체 생성 (학습 시 사용한 설정과 동일하게 유지: clipLimit=2.0, tileGridSize=(8, 8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ---------------------------------------------------------
    # 2. 스마트 역매핑 딕셔너리 만들기 (기존 로직 동일)
    # ---------------------------------------------------------
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        yolo_names = yaml.safe_load(f)['names']
        
    json_category_map = {} 
    original_json_names = {} 

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
                except: pass

    inverse_class_map = {}
    for yolo_id, yaml_name in yolo_names.items():
        clean_yaml = str(yaml_name).replace(" ", "")
        if clean_yaml in json_category_map:
            inverse_class_map[int(yolo_id)] = json_category_map[clean_yaml]
        else:
            possible_matches = difflib.get_close_matches(clean_yaml, json_category_map.keys(), n=1, cutoff=0.5)
            if possible_matches:
                inverse_class_map[int(yolo_id)] = json_category_map[possible_matches[0]]

    # ---------------------------------------------------------
    # 3. 모델 로드 및 추론 (CLAHE 변환 추가)
    # ---------------------------------------------------------
    model = YOLO(MODEL_PATH)
    
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int("".join(re.findall(r'\d+', x))))
    
    results_list = []
    ann_id_counter = 1 

    print(f"🚀 CLAHE 적용 후 총 {len(image_files)}장의 이미지 추론을 시작합니다.")
    for img_name in tqdm(image_files):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        image_id_str = "".join(re.findall(r'\d+', img_name))
        image_id = int(image_id_str) if image_id_str else 0
            
        # --- [추가] 실시간 CLAHE 전처리 ---
        img = cv2.imread(img_path)
        if img is None: continue
        
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        clahe_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        # -------------------------------

        # 파일 경로 대신 변환된 이미지 배열(clahe_img)을 직접 입력
        outputs = model.predict(source=clahe_img, conf=0.25, imgsz=640, device='mps', verbose=False)
        
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