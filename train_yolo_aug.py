from __future__ import annotations

"""
YOLO 학습과 validation 평가를 한 번에 수행하는 학습 스크립트.

설정 파일 또는 CLI 인자를 받아 모델을 학습한 뒤, 같은 실험의 `validation` 성능을
평가해 mAP 지표를 출력한다. 최종 성능 결과는 metrics JSON으로 저장해
`experiments.md` 기록 및 실험 비교의 기준값으로 사용한다.
"""

from pathlib import Path
import argparse
import datetime
import json
import re
import sys

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_ROOT / "metrics"

# 👉 runs 경로를 절대경로로 고정 (핵심)
RUNS_DIR = PROJECT_ROOT / "runs"


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def find_default_dataset_yaml(exp_name: str) -> Path:
    # 💡 터미널에서 입력받은 exp_name을 바탕으로 yaml 경로를 동적으로 찾습니다.
    expected_path = PROJECT_ROOT / "data" / "yolo_dataset" / exp_name / "dataset.yaml"
    if expected_path.exists():
        return expected_path

    candidates = sorted(PROJECT_ROOT.glob("data/**/dataset.yaml"))
    hint = ""
    if candidates:
        hint = "\n\nFound these candidates:\n" + "\n".join(f"- {p}" for p in candidates)

    raise FileNotFoundError(
        "dataset.yaml not found.\n"
        f"- looked for: {expected_path}"
        f"{hint}\n\n"
        f"If you intended to use the existing dataset, ensure you ran `prepare_yolo_dataset.py --exp_name {exp_name}` first.\n"
    )


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    else:
        raise ValueError(f"Unsupported config type: {config_path} (use .yaml/.yml or .json)")

    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping/dict, got: {type(data).__name__}")

    return data


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    # 💡 우리의 핵심 인자 추가!
    parser.add_argument(
        "--exp_name", 
        type=str, 
        default=defaults.get("exp_name", "baseline"), 
        help="실험 이름을 입력하세요 (예: baseline, oversampling, copy_paste_v1, copy_paste_v2)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=defaults.get("config"),
        help="Path to config file (.yaml/.yml/.json). CLI args override config values.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(defaults["data"]) if defaults.get("data") else None,
        help="Path to Ultralytics dataset.yaml (default: auto-detect based on exp_name).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=defaults.get("model", "yolo11s.pt"),
        help="Model path (e.g., yolo11s.pt, yolov8n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 50)), help="number of epochs")
    parser.add_argument("--imgsz", type=int, default=int(defaults.get("imgsz", 640)), help="image size")
    parser.add_argument("--batch", type=int, default=int(defaults.get("batch", 4)), help="batch size")
    parser.add_argument(
        "--name",
        type=str,
        default=defaults.get("name", None),
        help="experiment folder name (defaults to exp_name if not provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", ""),
        help="cuda device, i.e. 0 or mps or cpu",
    )
    # --- 증강(Augmentation) 제어용 인자 --- #
    parser.add_argument("--fliplr", type=float, default=float(defaults.get("fliplr", 0.5)))
    parser.add_argument("--flipud", type=float, default=float(defaults.get("flipud", 0.0)))
    parser.add_argument("--degrees", type=float, default=float(defaults.get("degrees", 0.0)))
    parser.add_argument("--mosaic", type=float, default=float(defaults.get("mosaic", 1.0)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 0)))
    parser.add_argument("--deterministic", action="store_true", default=bool(defaults.get("deterministic", True)))
    parser.add_argument("--copy_paste", type=float, default=float(defaults.get("copy_paste", 0.0)))
    parser.add_argument("--mixup", type=float, default=float(defaults.get("mixup", 0.0)))
    # --------------box, cls weights------------ #
    parser.add_argument("--box", type=float, default=7.5, help="box loss gain")
    parser.add_argument("--cls", type=float, default=0.5, help="cls loss gain")
    parser.add_argument("--dfl", type=float, default=1.5, help="dfl loss gain") # 👈 이거 추가!
    return parser


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=Path, default=None)
    known, _ = base.parse_known_args()

    defaults: dict = {}
    if known.config:
        defaults = load_config(known.config)
        defaults["config"] = known.config

        allowed_keys = {
            "exp_name",  # 💡 exp_name 키 허용
            "config",
            "data",
            "model",
            "epochs",
            "imgsz",
            "batch",
            "name",
            "device",
            "fliplr",
            "flipud",
            "degrees",
            "mosaic",
            "copy_paste",
            "mixup",
            "seed",
            "deterministic",
            # box, cls
            "box",
            "cls",
            "dfl",
        }
        unknown_keys = sorted(set(defaults.keys()) - allowed_keys)
        if unknown_keys:
            print(
                f"Warning: ignoring unknown config keys: {unknown_keys}",
                file=sys.stderr,
            )

    parser = build_parser(defaults)
    return parser.parse_args()


def infer_source_experiment(model_path: str) -> str | None:
    match = re.search(r"(?:^|[/_])exp[_-]?(\d+)(?:[^0-9]|$)", model_path, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"pill_exp(\d+)", model_path, flags=re.IGNORECASE)
    return f"exp{match.group(1)}" if match else None


def infer_model_name(model_path: str) -> str:
    match = re.search(r"(yolo\d+[a-z]+|yolov\d+[a-z]+|rtdetr[\w-]*)", model_path, flags=re.IGNORECASE)
    return match.group(1) if match else model_path


def main() -> None:
    args = parse_args()
    device = args.device or get_device()
    
    # 💡 run_name을 동적으로 설정 (명시적으로 --name을 안 줬으면 --exp_name을 사용)
    run_name = args.name if args.name else args.exp_name
    
    print(f"Using device: {device}")
    print(f"Experiment Name: {args.exp_name}")

    if args.config:
        print(f"Config: {args.config}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Runs dir: {RUNS_DIR}")

    # 💡 exp_name을 넘겨서 해당 실험의 dataset.yaml 경로를 가져옴
    data_yaml = args.data or find_default_dataset_yaml(args.exp_name)
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
        project=str(RUNS_DIR),
        name=run_name,          # 💡 runs/detect/{run_name} 형태로 예쁘게 저장됨
        pretrained=True,
        verbose=True,
        plots=True,
        save=True,

        fliplr=args.fliplr,
        flipud=args.flipud,
        degrees=args.degrees,
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        mixup=args.mixup,
        
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        cos_lr=True,
        seed=args.seed,
        deterministic=args.deterministic,

        # box, cls 가중치 추가
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
    )

    print("Training finished. Evaluating best model metrics...")
    
    val_results = model.val(
        data=str(data_yaml),
        split='val',
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,          # args.device 대신 판별된 device 사용
        plots=True
    )
    
    map50 = val_results.results_dict.get('metrics/mAP50(B)', 0.0)
    map75 = val_results.maps[5] if hasattr(val_results, 'maps') and len(val_results.maps) > 5 else 0.0
    map50_95 = val_results.results_dict.get('metrics/mAP50-95(B)', 0.0)
    
    precision = val_results.results_dict.get('metrics/precision(B)', 0.0)
    recall = val_results.results_dict.get('metrics/recall(B)', 0.0)
    
    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    metrics_payload = {
        "experiment": run_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "dataset_split": "val",
        "data": str(data_yaml),
        "model_path": args.model,
        "model_name": infer_model_name(args.model),
        "source_experiment": infer_source_experiment(args.model),
        "epoch": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "best_epoch": int(getattr(val_results, "epoch", args.epochs)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1_score),
        "mAP50": float(map50),
        "mAP75": float(map75),
        "mAP50-95": float(map50_95),
    }
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{run_name}_val_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 60)
    print(f"      EXPERIMENT REPORT: {run_name}")
    print("=" * 60)
    print(f" ➡️  Precision: {precision:.4f}")
    print(f" ➡️  Recall:    {recall:.4f}")
    print(f" ➡️  F1-Score:  {f1_score:.4f}")
    print(f" ➡️  mAP@50:    {map50:.4f}")
    print(f" ➡️  mAP@75:    {map75:.4f}")
    print(f" ➡️  mAP@50-95: {map50_95:.4f}")
    print(f" ➡️  saved:     {metrics_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()