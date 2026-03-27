# 🧪 Pill Detection Experiment Log
## 1. 🎯 Objective
- 경구약제 이미지에서 객체 검출(Object Detection) 수행
- YOLOv8 기반 모델을 사용하여 성능 최적화
- 단순 mAP 향상이 아닌 실제 추론 품질 개선 목표
---
## 2. 🧱 Dataset
- 총 이미지 수: 약 800장
- Annotation: COCO format (bbox: [x, y, w, h])
- 클래스 수: 다수 (알약 종류별)
---
## 3. ⚙️ Preprocessing
### 3.1 기본 전처리
- annotation 병합 (image-level)
- invalid/missing bbox 제거
- label map 생성

### 3.2 Train/Validation Split
- 초기: baseline 비교를 위해 random 8:2 split 적용
- 이후: 계층적 분할 (stratified split)

### 3.3 Rare Class 처리
- rare class 기준: 객체 수 ≤ 5
- rare class 포함 이미지 oversampling
---
## 4.  📊 Training Results
### 🔬 **Exp1. Baseline (계층적 분할 + YOLOv8n)**
**설정**
- 모델: YOLOv8n (Nano)
- 데이터 분할: 다중 계층적 분할
- 전처리: invalid / missing bbox 처리
- 학습: epochs=50, imgsz=960
<br/>

**결과**
- mAP50: 0.8697
- mAP50-95: 0.8414
- Precision: 0.9009
- Recall: 0.7439
<br/>

### 🔬 **Exp2. Rare Class Oversampling**
**변경점**
- rare class 포함 이미지 oversampling
- 나머지 설정 동일 (YOLOv8n 유지)
<br/>

**결과**
- mAP50 ≈ 0.87
- mAP50-95 ≈ 0.84
- Precision ≈ 0.90+
- Recall ≈ 0.74
<br/>

### 🔬 **Exp3. YOLOv8s 모델 확장**
**변경점**
- 모델: YOLOv8n → YOLOv8s
- rare class oversampling 유지
<br/>

**결과**
- mAP50 ≈ 0.99
- mAP50-95 ≈ 0.97
- mAP75-95 ≈ 0.95
- Precision ≈ 0.94 ~ 0.96
- Recall ≈ 0.97+
<br/>

### 🔬 **Exp4. Hard Example Mining (HEM)**
**방법**
1. best.pt로 train 이미지 재추론
2. GT와 비교하여 hard image 선정
- missed_gt
- low_conf
- false_positive
3. 해당 이미지 train에 복제
<br/>

**결과**
- Num hard images: 5 -> 복제 후: 10장
- 이미 모델 성능이 충분히 높은 상태이며, hard sample 비율이 너무 적어서 성능 변화 거의 없음.
<br/>

### 🔬 **Exp5. YOLOv8m + Augmentation**
**변경점**
- 모델: YOLOv8m
- imgsz: 640
- augmentation:
    - mosaic=1.0
    - mixup=0.1
    - degrees=15
- optimizer: AdamW
<br/>

**결과**
- mAP50: 0.838
- mAP50-95: 0.747
- Precision: 0.905
- Recall: 0.824
- 성능 하락 원인:
    - imgsz 감소 (960 → 640)
    - augmentation 과도 적용
    - 학습 안정성 저하
<br/>

### 🔬 **Exp6. YOLOv8m 개선 (고해상도 + box weight + TTA)**
**변경점**
- imgsz: 960 (복구)
- box loss weight 증가 (box=15.0)
- augmentation 유지
- TTA 적용
<br/>

**결과**
- mAP@[0.75:0.95] ≈ 0.95199
- Precision ≈ 0.94 ~ 0.96
- Recall ≈ 0.97+
<br/>

### **Overall Comparison**
| Experiment | Model | Key Change        | mAP50    | mAP50-95 | Recall    |
| ---------- | ----- | ----------------- | -------- | -------- | --------- |
| Exp1       | v8n   | baseline          | 0.87     | 0.84     | 0.74      |
| Exp2       | v8n   | oversampling      | ~same    | ~same    | ~same     |
| Exp3       | v8s   | model upgrade     | **0.99** | **0.97** | **0.97+** |
| Exp4       | v8s   | HEM               | no gain  | no gain  | no gain   |
| Exp5       | v8m   | aug + low res     | ↓        | ↓        | ↑         |
| Exp6       | v8m   | high res + tuning | ↑        | ↑        | ↑         |
---
