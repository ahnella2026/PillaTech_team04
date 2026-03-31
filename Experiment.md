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
## 4. 💡 각 실험별 핵심 가설 (Hypothesis)
### [초기 탐색 및 기반 구축]
- **Exp1 (Baseline):** 모델 성능 측정의 기준점을 설정하고, 다중 계층적 분할을 통해 데이터 편향 없는 객관적인 평가 환경을 구축한다.
- **Exp2 (Oversampling):** 소수 클래스의 노출 빈도를 높이면 모델이 희귀 알약의 특징을 더 잘 학습하여 전체적인 재현율(Recall)이 상승할 것이다.
- **Exp3 (Model Upgrade):** 파라미터 수가 더 많은 YOLOv8s 모델을 사용하면, 알약의 미세한 각인이나 형태적 특징(Feature)을 더 정교하게 추출할 수 있을 것이다.
<br/>

### [성능 고도화 및 시행착오]
- **Exp4 (HEM)**: 모델이 틀린 어려운 샘플(Hard Example)을 집중 학습시키면 경계선에 있는 모호한 객체에 대한 판별력이 향상될 것이다.
- **Exp5 (Augmentation)**: Mosaic, Mixup 등 강력한 증강 기법을 도입하면 모델의 일반화 성능이 좋아질 것이나, 저해상도(640) 환경에서는 작은 알약의 특징 소실이 발생할 수 있다.
- **Exp6 (Optimization)**: 해상도를 복구하고 Box Loss 가중치를 높이면 객체의 위치 정보를 더 정밀하게 학습하여 mAP@[0.75:0.95]와 같은 고정밀도 지표가 개선될 것이다.
<br/>

### [Baseline 2.0: 최신 아키텍처 및 전처리 도입]
- **Exp7 (v11s)**: 최신 YOLOv11 아키텍처를 도입하면 더 적은 파라미터로도 효율적인 연산을 수행하며, 640 해상도에서도 이전 세대 고해상도(960) 이상의 성능을 낼 것이다.
- **Exp8 (CLAHE)**: 조명 불균형이 심한 알약 이미지에 CLAHE 전처리를 적용하면, 배경과 알약의 대비가 명확해지고 표면 각인 정보가 강조되어 정밀도가 상승할 것이다.
- **Exp9 (Copy-Paste)**: 배경 이미지 위에 희귀 클래스 객체를 합성하는 Copy-Paste 증강을 적용하면, 다양한 배경 맥락에서 객체를 학습하게 되어 미탐지(False Negative)를 줄이고 재현율을 극대화할 수 있을 것이다.
---
## 5.  📊 Training Results
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
**설정**
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

### 🔬 **Baseline 2.0 이후**
<br/>

### 🔬 **Exp7. Baseline 2.0 모델 해상도 실험**
**변경점**
- 모델: YOLOv11s.py
- 데이터 분할: 다중 계층적 분할
- 전처리: invalid / missing box 처리, rare class 포함 이미지 oversampling
- 학습: epochs=50, imgsz=640, batch_size=16
<br/>

**결과**
- mAP50: 0.99409
- mAP50-95: 0.98723
- Precision: 0.97356
- Recall: 0.99146
<br/>

### 🔬 **Exp8. Baseline 2.0 + CLAHE**
**변경점**
- 데이터 전처리/증강: CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
- 나머지 설정 동일 (YOLOv11s 유지)
<br/>

**결과**
- mAP50: 0.9948
- mAP50-95: 0.9907
- Precision: 0.9768
- Recall: 0.9911
- F1-Score: 0.9839
<br/>

### 🔬 **Exp9. Exp8 + Copy-Paste Augmentation**
**변경점**
- 데이터 증강: Copy-Paste (희귀 클래스 객체 합성) 추가
- 나머지 설정 동일 (CLAHE 유지)
<br/>

**결과**
- mAP50: 0.9931
- mAP50-95: 0.9867
- Precision: 0.9721
- Recall: 0.9930
- F1-Score: 0.9824
<br/>

### **Overall Comparison**
| Experiment | Model | Key Change         | mAP50    | mAP50-95 | Recall    |
| ---------- | ----- | -----------------  | -------- | -------- | --------- |
| Exp1       | v8n   | baseline           | 0.87     | 0.84     | 0.74      |
| Exp2       | v8n   | oversampling       | ~same    | ~same    | ~same     |
| Exp3       | v8s   | model upgrade      | 0.99     | 0.97     | 0.97+     |
| Exp4       | v8s   | HEM                | no gain  | no gain  | no gain   |
| Exp5       | v8m   | aug + low res      | ↓        | ↓        | ↑         |
| Exp6       | v8m   | high res + tuning  | ↑        | ↑        | ↑         |
| Exp7       | v11s  | Baseline 2.0 (Res) | 0.9941   | 0.9872   | 0.9915    |
| Exp8       | v11s  | + CLAHE            |**0.9948**|**0.9907**|**0.9911** |
| Exp9       | v11s  | + Copy-Paste       | 0.9931   | 0.9867   | 0.9930    |
---
<br/>

,,
"<img src=""./runs/pill_exp_clahe_copy_paste/predict_result_final/1.png"" width=""250"">","<img src=""./runs/pill_exp_clahe_copy_paste/predict_result_final/2.png"" width=""250"">","<img src=""./runs/pill_exp_clahe_copy_paste/predict_result_final/3.png"" width=""250"">"
Sample 01,Sample 02,Sample 03
"<img src=""./runs/pill_exp_clahe_copy_paste/predict_result_final/4.png"" width=""250"">","<img src=""./runs/pill_exp_clahe_copy_paste/predict_result_final/5.png"" width=""250"">","<img src=""./runs/pill_exp_clahe_copy_paste/predict_result_final/6.png"" width=""250"">"
Sample 04,Sample 05,Sample 06
