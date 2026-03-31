import cv2
import os
import glob
from tqdm import tqdm

def apply_clahe_v3(base_path, new_path):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # 원본 구조: images/train, images/val
    sub_dirs = ['train', 'val'] 
    
    for sub in sub_dirs:
        # 1. 원본 이미지 경로 설정 (data/yolo_dataset/images/train)
        input_dir = os.path.join(base_path, "images", sub)
        img_list = glob.glob(os.path.join(input_dir, "*.png"))
        
        if not img_list:
            print(f"⚠️ {input_dir}에서 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
            continue

        # 2. 저장될 경로 설정 (data/yolo_cleaned_clahe/seed_777/train/images)
        output_dir = os.path.join(new_path, sub, "images")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📷 {sub} 데이터 변환 중... ({len(img_list)}개)")
        for img_path in tqdm(img_list):
            img = cv2.imread(img_path)
            if img is None: continue
            
            # CLAHE 적용
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = clahe.apply(yuv[:,:,0])
            clahe_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # 저장
            save_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, clahe_img)

# 설정값
BASE_DATA_PATH = 'data/yolo_dataset' # 원본 데이터 루트
NEW_DATA_PATH = 'data/yolo_cleaned_clahe/seed_777' # 새 데이터 루트

apply_clahe_v3(BASE_DATA_PATH, NEW_DATA_PATH)
print("✅ CLAHE 변환 및 폴더 정리 완료!")