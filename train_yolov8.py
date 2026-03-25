from __future__ import annotations

from pathlib import Path
import shutil

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"

# 👉 runs 경로를 절대경로로 고정 (핵심)
RUNS_DIR = PROJECT_ROOT / "runs"


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Runs dir: {RUNS_DIR}")

    # 👉 YOLO 모델 로드
    model = YOLO("yolov8s.pt")

    # 👉 학습
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=960,
        batch=4,
        device=device,

        # ⭐ 핵심: 절대경로 사용
        project=str(RUNS_DIR),

        # 실험 이름
        name="pill_yolov8s_v1",

        pretrained=True,
        verbose=True,
    )

    print("Training finished.")


if __name__ == "__main__":
    main()