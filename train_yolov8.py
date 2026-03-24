from __future__ import annotations

from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML_LEGACY = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"
DEFAULT_DATA_YAML_CLEANED = (
    PROJECT_ROOT / "data" / "yolo_cleaned" / "seed_777" / "dataset.yaml"
)

# 👉 runs 경로를 절대경로로 고정 (핵심)
RUNS_DIR = PROJECT_ROOT / "runs"


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def find_default_dataset_yaml() -> Path:
    if DEFAULT_DATA_YAML_CLEANED.exists():
        return DEFAULT_DATA_YAML_CLEANED
    if DEFAULT_DATA_YAML_LEGACY.exists():
        return DEFAULT_DATA_YAML_LEGACY

    candidates = sorted(PROJECT_ROOT.glob("data/**/dataset.yaml"))
    hint = ""
    if candidates:
        hint = "\n\nFound these candidates:\n" + "\n".join(f"- {p}" for p in candidates)

    raise FileNotFoundError(
        "dataset.yaml not found.\n"
        f"- looked for: {DEFAULT_DATA_YAML_CLEANED}\n"
        f"- looked for: {DEFAULT_DATA_YAML_LEGACY}"
        f"{hint}\n\n"
        "If you intended to use the existing cleaned dataset, pass:\n"
        "  python train_yolov8.py --data data/yolo_cleaned/seed_777/dataset.yaml\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to Ultralytics dataset.yaml (default: auto-detect).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model weights or model name (e.g. yolov8n.pt, yolov8s.pt, path/to.pt).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="pill_exp1_stratified_full")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cpu, 0, mps). Default: auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_device()
    print(f"Using device: {device}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Runs dir: {RUNS_DIR}")

    data_yaml = args.data or find_default_dataset_yaml()
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    # 👉 YOLO 모델 로드
    model = YOLO(args.model)

    # 👉 학습
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,

        # ⭐ 핵심: 절대경로 사용
        project=str(RUNS_DIR),

        # 실험 이름
        name=args.name,

        pretrained=True,
        verbose=True,

        # === 실험 1: Baseline 증강 유지 ===
        fliplr=0.5,             # YOLOv8 기본값
        flipud=0.0,
        degrees=0.0,
        mosaic=1.0,
        
        # === 최적화 설정 (기본값 위주) ===
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        cos_lr=True,
    )

    print("Training finished. Evaluating best model metrics...")
    
    # [Final Validation for detailed metrics]
    # This evaluates the 'best.pt' model found in the results directory.
    val_results = model.val(
        data=args.data,
        split='val',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True
    )
    
    # Extract map50, map75, map50-95
    # maps[0] is mAP@50, maps[5] is mAP@75, ..., maps[9] is mAP@95
    map50 = val_results.results_dict['metrics/mAP50(B)']
    map75 = val_results.maps[5]
    map50_95 = val_results.results_dict['metrics/mAP50-95(B)']
    
    print("\n" + "=" * 60)
    print(f"      EXPERIMENT REPORT: {args.name}")
    print("=" * 60)
    print(f" ➡️  mAP@50:    {map50:.4f}")
    print(f" ➡️  mAP@75:    {map75:.4f}")
    print(f" ➡️  mAP@50-95: {map50_95:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
