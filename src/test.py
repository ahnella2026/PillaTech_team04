import os
import pandas as pd
import re
from ultralytics import YOLO
from tqdm import tqdm

def run_test_and_save_csv():
    # 1. 경로 설정 (사용자 환경에 맞게 수정)
    # 모델 경로
    MODEL_PATH = 'runs/detect/PillaTech/exp_high_res/weights/best.pt'
    # test 이미지 경로
    TEST_IMG_DIR = 'data/sprint_ai_project1_data/test_images' 
    # 예측결과 csv 파일명
    OUTPUT_CSV = 'pill_detection_results.csv'
    
    # 2. 모델 로드
    model = YOLO(MODEL_PATH)
    
    # 3. 이미지 파일 목록 확보 및 정렬
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort() # 순서대로 처리하기 위함
    
    results_list = []
    ann_id_counter = 1 # 고유한 annotation_id를 위한 카운터
    
    print(f"🚀 총 {len(image_files)}장의 이미지에 대해 검출을 시작합니다.")

    for img_name in tqdm(image_files):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        # 🔍 파일명에서 숫자만 추출하여 image_id 생성
        # 예: '1.png' -> 1
        image_id_str = "".join(re.findall(r'\d+', img_name))
        image_id = int(image_id_str) if image_id_str else 0
            
        # 모델 예측 (고해상도 설정 유지)
        outputs = model.predict(source=img_path, conf=0.25, imgsz=1024, verbose=False)
        
        for r in outputs:
            boxes = r.boxes
            for box in boxes:
                # YOLO ID를 원본 Category ID로 변환 (Phase 1의 class_map 역순)
                # 만약 변환이 필요 없다면 yolo_cls_id를 그대로 사용하세요.
                yolo_cls_id = int(box.cls[0]) 
                score = float(box.conf[0])
                
                # BBox 좌표 변환 [x1, y1, x2, y2] -> [x, y, w, h]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_x = int(x1)
                bbox_y = int(y1)
                bbox_w = int(x2 - x1)
                bbox_h = int(y2 - y1)
                
                # 결과 리스트 추가 (1개 객체당 1개의 로우)
                results_list.append({
                    'annotation_id': ann_id_counter, # 전체 로우 중 고유값
                    'image_id': image_id,           # 파일명의 숫자
                    'category_id': yolo_cls_id,      # 클래스 번호
                    'bbox_x': bbox_x,
                    'bbox_y': bbox_y,
                    'bbox_w': bbox_w,
                    'bbox_h': bbox_h,
                    'score': round(score, 3)         # 신뢰도 점수
                })
                ann_id_counter += 1

    # 4. 데이터프레임 생성 및 CSV 저장
    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ 분석 완료! 총 {len(df)}개의 객체가 검출되었습니다.")
    print(f"📄 결과 파일: {os.path.abspath(OUTPUT_CSV)}")
    print(df.head()) # 상위 결과 미리보기

if __name__ == '__main__':
    run_test_and_save_csv()