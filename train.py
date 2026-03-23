import os
import shutil
from ultralytics import YOLO
from dotenv import load_dotenv
from pathlib import Path

# 1. 환경 변수 로드 (.env 설정값 가져오기)
load_dotenv()

def run_training():
    # .env에서 설정값 읽기
    seeds_str = os.getenv("RANDOM_SEEDS", "42,123,777")
    seeds = [int(s.strip()) for s in seeds_str.split(",")]
    
    epochs = int(os.getenv("TRAIN_EPOCHS", "50"))
    imgsz = int(os.getenv("TRAIN_IMG_SIZE", "640"))
    output_base = os.getenv("DATA_OUTPUT_DIR", "./data/yolo_cleaned")

    print(f"\n[INFO] 학습 시작 예정 - Seeds: {seeds}, Epochs: {epochs}, Imgsz: {imgsz}")

    # 2. 각 시드별로 자동 순차 학습
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"STEP: Seed {seed} 학습 시작")
        print(f"{'='*50}")

        # 모델 초기화 (이미 존재하는 가중치 사용)
        model = YOLO("yolov8n.pt")

        # 시드별 데이터 설정 파일 경로
        yaml_path = Path(output_base) / f"seed_{seed}" / "dataset.yaml"

        # 학습 실행 (Ultralytics API 사용)
        model.train(
            data     = str(yaml_path),
            epochs   = epochs,
            imgsz    = imgsz,
            project  = "runs/baseline",
            name     = f"seed_{seed}",
            seed     = seed,  # 모델 학습 시드 고정
            verbose  = True
        )

        # 학습 완료 후 'Best' 가중치만 최상위 weights 폴더로 추출
        weights_export_dir = Path("weights")
        weights_export_dir.mkdir(exist_ok=True, parents=True)
        
        # YOLOv8이 기본적으로 저장하는 output 폴더 경로 구성
        best_source = Path("runs/baseline") / f"seed_{seed}" / "weights" / "best.pt"
        target_destination = weights_export_dir / f"baseline_best_seed_{seed}.pt"

        if best_source.exists():
            shutil.copy2(best_source, target_destination)
            print(f"[SUCCESS] 최적 가중치 추출 완료: {target_destination}")
        else:
            print(f"[WARNING] 학습은 끝났지만 가중치 파일을 찾을 수 없습니다: {best_source}")

if __name__ == "__main__":
    run_training()
