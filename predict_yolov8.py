from __future__ import annotations

from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "runs" / "pill_yolov8s_v1" / "weights" / "best.pt"
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "test_images"
RUNS_DIR = PROJECT_ROOT / "runs" / "pill_yolov8s_v1"

# macOS 기본 한글 폰트 예시
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def shorten_label(label: str, max_len: int = 12) -> str:
    if len(label) <= max_len:
        return label
    return label[:max_len] + "..."


def draw_korean_text(
    img_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int = 24,
) -> np.ndarray:
    """
    OpenCV 이미지 위에 PIL로 한글 텍스트 그리기
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.truetype(FONT_PATH, font_size)

    # text bbox
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left, top, right + 6, bottom + 4), fill=(255, 255, 255))
    draw.text((x + 3, y), text, font=font, fill=(0, 0, 0))

    img_rgb = np.array(pil_img)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    device = get_device()

    print(f"Using device : {device}")
    print(f"Model path   : {MODEL_PATH}")
    print(f"Source dir   : {SOURCE_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source path not found: {SOURCE_DIR}")

    model = YOLO(str(MODEL_PATH))

    output_dir = RUNS_DIR / "predict_pill_yolov8s_v1_shortlabel_kor"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in SOURCE_DIR.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=0.1,
            imgsz=960,
            save=False,
            device=device,
            verbose=False,
        )

        result = results[0]
        img = cv2.imread(str(image_path))

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label = result.names[cls_id]
            short_label = shorten_label(label)
            text = f"{short_label} {conf:.2f}"

            # bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # 한글 텍스트
            text_y = max(0, y1 - 30)
            img = draw_korean_text(img, text, x1, text_y, font_size=24)

        save_path = output_dir / image_path.name
        cv2.imwrite(str(save_path), img)

    print("Prediction finished.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()