import os
import pandas as pd
import re
import yaml
import json
import difflib 
from ultralytics import YOLO
from tqdm import tqdm

def run_test_and_save_csv():
    # 1. 경로 설정 (사진에 맞게 경로 수정 완료)
    MODEL_PATH = 'runs/pill_yolov8n_v2/weights/best.pt'
    TEST_IMG_DIR = 'data/sprint_ai_project1_data/test_images' 
    OUTPUT_CSV = 'basemodel_results.csv'
    YAML_PATH = 'data/yolo_dataset_v2_stratified/dataset.yaml'
    JSON_DIR = 'data/sprint_ai_project1_data/train_annotations' 

    # ---------------------------------------------------------
    # 2. 스마트 역매핑 딕셔너리 만들기 (difflib 유사도 검사)
    # ---------------------------------------------------------
    MANUAL_OVERRIDE = {}

    # A. YAML 파일 읽기
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        yolo_names = yaml.safe_load(f)['names']
        
    json_category_map = {} 
    original_json_names = {} 

    print(f"🔍 '{JSON_DIR}' 폴더 탐색을 시작합니다. (시간이 조금 걸릴 수 있습니다)")
    
    # B. JSON 폴더 깊은 곳까지 싹 다 뒤져서 정보 수집
    for root, dirs, files in os.walk(JSON_DIR):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'categories' in data:
                            for cat in data['categories']:
                                # 공백을 1차적으로 모두 제거해서 저장
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
        # 수동 지정한 게 있다면 가장 먼저 적용
        if yaml_name in MANUAL_OVERRIDE:
            inverse_class_map[int(yolo_id)] = MANUAL_OVERRIDE[yaml_name]
            continue
            
        clean_yaml = str(yaml_name).replace(" ", "")
        
        # 1단계: 띄어쓰기 뺀 이름이 완벽히 똑같은가?
        if clean_yaml in json_category_map:
            inverse_class_map[int(yolo_id)] = json_category_map[clean_yaml]
        else:
            # 2단계: 똑같지 않다면, 가장 글자가 비슷한 알약 이름을 JSON에서 찾아라! (유사도 50% 이상)
            possible_matches = difflib.get_close_matches(clean_yaml, json_category_map.keys(), n=1, cutoff=0.5)
            
            if possible_matches:
                best_match = possible_matches[0]
                matched_id = json_category_map[best_match]
                inverse_class_map[int(yolo_id)] = matched_id
                print(f"💡 자동 보정 성공: '{yaml_name}' ➡️ JSON의 '{original_json_names[best_match]}' (ID: {matched_id})")
            else:
                unmapped_classes.append(yaml_name)

    # 매핑 결과 리포트
    if unmapped_classes:
        print("\n🚨 여전히 짝을 찾지 못한 클래스들입니다:")
        for name in unmapped_classes:
            print(f" - {name}")
        print("👉 이 알약들은 JSON 파일에 정보가 아예 없거나 너무 다릅니다. 코드 상단의 MANUAL_OVERRIDE에 직접 ID를 적어주세요!\n")
    else:
        print("\n🎉 모든 56개 알약의 클래스 매핑이 완벽하게 완료되었습니다!\n")

    # ---------------------------------------------------------
    # 3. 모델 로드 및 추론
    # ---------------------------------------------------------
    model = YOLO(MODEL_PATH)
    
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int("".join(re.findall(r'\d+', x))))
    
    results_list = []
    ann_id_counter = 1 

    print(f"🚀 총 {len(image_files)}장의 이미지에 대해 검출을 시작합니다.")
    for img_name in tqdm(image_files):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        image_id_str = "".join(re.findall(r'\d+', img_name))
        image_id = int(image_id_str) if image_id_str else 0
            
        outputs = model.predict(source=img_path, conf=0.25, imgsz=1024, verbose=False)
        
        for r in outputs:
            boxes = r.boxes
            for box in boxes:
                yolo_cls_id = int(box.cls[0]) 
                score = float(box.conf[0])
                
                # 매핑된 ID 적용 (없으면 -1)
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