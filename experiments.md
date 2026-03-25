# Pill Detection Experiment Log

## 📑 지표 정의 및 측정 이유 (Metric Definitions & Rationale)

1.  **mAP@50 (Detection 성공률)**: "일단 찾기는 했는가?" 
    *   가장 기본적이고 너그러운 기준임. 알약의 존재를 놓치지는 않았는지를 판단하는 기초 실력입니다.
2.  **mAP@75 (Localization 정밀도)**: **"칼같이 찾았는가?"** 
    *   알약 테두리를 얼마나 정확하게 감쌌는지를 봄. 초급 프로젝트처럼 **알약의 각인(문자/로고)을 분석해야 하는 작업**이 뒤따른다면, 박스 오차가 작아야 글자가 잘리지 않고 정확한 OCR이 가능해짐
3.  **mAP@50-95 (종합 실력)**: "전체적인 품질이 어떤가?" 
    *   너그러운 기준(@50)부터 아주 깐깐한 기준(@95)까지의 평균임. 모델의 전반적인 완성도를 나타내는 가장 신뢰할 만한 점수임.
4.  **Best Epoch (최적화 지점)**: "언제 더 이상 학습할 필요가 없었는가?" 
    *   최대 성능을 낸 시점을 알아야 불필요한 과적합(Overfitting) 여부를 판단하고 학습 속도를 조절할 수 있음. 


## 📊 실험 결과 요약 (Experiment Summary Table)

| ID | 실험명 | 분기 조건 (Split / Augment) | Model | Res | Epoch | **mAP50** | **mAP@50-95** | **mAP@75** | Best Epoch | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **캐글** | **캐글용** | Stratified Split/욜로기본증강설정 | v8n | 640 | 50 | 50 | - | - | 0.79 | Kaggle: 0.70166 |
| **Ref 1**| **예원** | Random Split / 욜로기본증강설정 | v8n | 640 | 50 | **0.790** | 0.760 | - | - | 예원님 결과 |
| **Exp 0** | **YOLOv8n** | Random Split / 욜로기본증강설정 | v8n | 640 | 50 | 0.959 | 0.935 | **0.950** | 50 | • 극단적 데이터 누수 발생 <br>• 캐글 베이스라인 실험을 내 컴퓨터에서 실험한 결과 |
| **Exp 1-1** | **Stratified** | YOLO 기본증강 + Stratified | v8n | 640 | 20 | 0.402 | 0.384 | **0.395** | 20 | 계층적 분할 시작 |
| **Exp 1-2**| **Stratified** | YOLO 기본증강 + Stratified | v8n | 640 | 50 | **0.960** | 0.943 | 0.945 | **36** | 누수 잔존 확인 |
| **Exp 2** | **v11s Pivot** | YOLO 기본증강 + Stratified | **v11s** | 640 | 50 | **0.994** | **0.990** | **0.995** | **50** | 모델 교체 |
| **Exp 3** | **Hi-Res** | YOLO 기본증강 + Stratified | **v11s** | 960 | 50 | **0.995** | **0.990** | **0.995** | **50** | 해상도 증량 |
| **Exp 4** | **Flip-Off** | YOLO 기본증강 (Flip Off) | **v11s** | 960 | 50 | **0.995** | **0.992** | **0.995** | **50** | 각인 보호 효과 |
| **Exp 5** | **Copy-Paste** | YOLO 기본증강 (Flip Off) + CP | **v11s** | 960 | 50 | **0.995** | **0.993** | **0.995** | **50** | **Kaggle 0.968** |
| **Exp 6** | **Rotation** | YOLO 기본 (Flip Off) + CP + Rotation | **v11s** | 960 | 50 | 0.990 | 0.940 | 0.995 | 50 | **Kaggle 0.854** |
| **Exp 7** | **Final-Res** | YOLO 기본 (Flip Off) + CP (No Rot) | **v11s** | 1024| 50 | **0.995** | **0.993** | **0.995** | **50** | 성능 정체 |
| **Exp 8** | **Dynamic NMS** | Exp5(best) 기준 NMS 튜닝 | v11s | 1024 | - | **0.9941** | **0.9926** | **0.9941** | - | iou=0.70에서 0.0001 하락, 최종 conf=0.20, iou=0.60 고정 |
| **Exp 9** | **Snapshot WBF** | Exp 5(best) 3-Epoch WBF | v11s | 1024 | - | - | - | - | - | 진행 예정 |
| **Exp 10** | **Multi-scale** | 800/960/1024 MS 앙상블 | v11s | Mix | - | - | - | - | - | 진행 예정 |
| **Exp 11** | **3-Seed Ens** | Seed (42,123,777) 훈련 | v11s | 1024 | 50 | - | - | - | - | 진행 예정 |
| **Exp 12** | **v26n Base** | 2026.01 NMS-free 아키텍처 | v26n | 1024 | 50 | - | - | - | - | 진행 예정 |
| **Exp 13** | **DETR Base** | Transformer 기반 모델 | DETR | 1024 | 50 | - | - | - | - | 진행 예정 |
| **Exp 14** | **Hetero WBF** | v11s + v26n + DETR 융합 | Mix | Mix | - | - | - | - | - | 진행 예정 |
| **Exp 15** | **Pseudo-L** | Test 셋 Pseudo-label 재학습 | Mix | Mix | - | - | - | - | - | 진행 예정 |
| **Exp 16** | **Colab Pro** | v11m, v12-S/M 고체급 학습 | Mix | 1024 | 50 | - | - | - | - | 진행 예정 |


---

## 🔬 정성적 분석 및 이슈 사항 (Insights & Issues)

### [우선순위 기반] 성능 최적화 후속 전략 (Post-Exp8 Strategy)
Exp 7(1024px 리사이즈) 실험 이후, 훈련 비용이 큰 모델 업그레이드는 보류하고 **캐글 실전 제출 횟수(하루 2회)를 효율적으로 사용하기 위한 단기/고도화 기법**을 최우선적으로 사용하기로 함. 

1.  **🔥 1순위 (재학습 없는 최적화 - 즉시 실행)** 
    *   **NMS 튜닝 (Exp 8, 완료)**: Validation 기준 최종값 `conf=0.20`, `iou=0.60` 확정 (`iou=0.70`은 0.0001 하락).
    *   **Snapshot WBF (Exp 9)**: 훈련된 최종 모델(Exp 5)의 마지막 3개 Epoch 무손실 가중치 결합 (리스크 0).
    *   **Multi-scale WBF (Exp 10)**: 기본 모델을 다양하게 리사이즈(800, 960, 1024)하여 WBF로 추론 결합.
2.  **🟡 2순위 (아키텍처 다변화 및 강건성 확보 - 병렬 진행)**
    *   **3-seed 훈련 (Exp 11)**: 모델의 우연성 분산을 제어하기 위해 3-seed(42, 123, 777) 교차 검증 적용.
    *   **YOLO26-n & RT-DETR (Exp 12, 13)**: 26년 최신 NMS-free 모델과 Transformer 아키텍처를 동원해 시각적 편향을 완전 다르게 가져감.
3.  **🔵 3순위 (최종 병기 - 앙상블 및 고급기법)**
    *   **Hetero WBF (Exp 14)**: 각기 다른 장점을 가진 YOLO와 RT-DETR의 예측값을 WBF로 융합하여 mAP 극대화.
    *   **Pseudo-Labeling (Exp 15)**: 가장 강력하게 앙상블된 결과를 Test셋의 레이블로 삼아 추가 재학습 진행.
4.  **💎 번외 (Colab Pro 유료 환경 활용 - A100 GPU 권장)**
    *   **고체급 모델 부스팅 (Exp 16)**: 현재 로컬(RTX 3060 12GB) 환경에서는 OOM(Out of Memory)으로 학습 불가능한 `YOLO11m` 모델이나, 계산량이 많아 CPU/저가형 GPU에서 느린 **2025년 최신 `YOLOv12-S/M` (Attention 기반 고성능 모델)**을 학습시킵니다. Colab Pro의 고용량 GPU(A100/V100)를 대여하면 1024px 이상의 고해상도 환경에서 높은 Batch-size로 압도적인 성능 점프가 가능할 것으로 기대하고 있습니다. 


### [Exp 0] Baseline (YOLOv8n / 640px)
*   **분석**: mAP가 0.959라는 비정상적인 결과 도출. 이는 유사도가 높은 사진들이 학습/검증 셋에 골고루 섞여 검증 데이터가 유출된 **Data Leakage(데이터 누수)** 현상임.
*   **교훈**: 랜덤 분할은 지표를 지나치게 낙관적으로 만들며, 실제 대회(Kaggle: 0.701) 성적과 극심한 괴리를 발생시킴을 확인.

### [Exp 1] Stratified Split (YOLOv8n / 640px / 20ep)
*   **분석**: 640px 해상도에서 랜덤 분할 대비 지표가 0.402로 급감함. 이는 희귀 클래스들이 검증 데이터셋에 정직하게 포함되면서 나타나는 "정상적인 지표 하락" 현상임.
*   **결정**: 0.402를 기준점으로 잡고 다음 실험 진행.

### [Exp 1-2] Stratified Split (YOLOv8n / 640px / 50ep)
*   **현상**: 640px 환경에서 50 Epoch까지 학습 시 mAP50이 다시 0.96까지 상승함.
*   **분석**: 계층적 분할을 하더라도 **동일 세션에서 촬영된 이미지들 간의 높은 유사성(Session Similarity)** 때문에 모델이 검증 셋을 '쉽게' 맞추는 경향이 여전히 존재함.
*   **한계**: 단순히 데이터를 나누는 것(Split Strategy)만으로는 데이터 불균형과 리큐지 문제를 완벽히 해결할 수 없음을 시사함.
*   **결정**: 모델의 일반화 성능(Generalization) 및 강건성(Robustness) 향상을 위한 Copy-Paste 증강 기법 도입이 절대적으로 필요함.

### [Exp 2] YOLO11 Transition (YOLO11s / 640px)
*   **분석**: 모델을 YOLOv8n에서 최신형인 YOLOv11s로 교체(640px 유지)한 결과, **mAP75 기준 0.995**라는 압도적인 수치를 기록함.
*   **통찰**: YOLOv11의 개선된 아키텍처가 특징 추출 및 정밀한 Localization에서 뛰어난 피지컬을 보여줌을 입증함.
*   **주의**: 0.99라는 수치는 현재의 검증 환경(Data Leakage) 내에서 이미 '학습의 정점'에 도달했음을 의미하며, 이 상태로 캐글에 제출할 경우 'Overfitting to Session' 이슈로 인해 점수 하락이 예상됨.
*   **결정**: 모델 성능은 확인되었으므로, 이제 인위적인 데이터 변주(Copy-Paste)가 주어진 상황에서 모델 실험을 해야 함  

### [Exp 3] High-Res Pivot (YOLO11s / 960px)
*   **분석**: 이미지 해상도를 640px에서 960px로 대폭 키워 학습했음에도, mAP 지표는 기존 640px(Exp 2)와 거의 동일한 **한계점(Ceiling)**에 부딪힘.
*   **통찰**: 알약의 각인 등 세부 특징을 더 명확하게 제공 하더라도, '기본적인 배경이 동일한 쌍둥이 이미지'라는 데이터셋 자체의 치명적 결함(데이터 누수)을 넘어서지는 못함.

### [Exp 4] No-Flip Fix (YOLO11s / 960px)
*   **분석**: 각인 보호를 위해 욜로 기본 설정인 fliplr=0.0(좌우 반전 금지) 조치 후 학습한 결과, mAP@50-95 지표가 0.989에서 0.992로 미세하게 상승함. 
*   **통찰**: 엉망이 된 오염 데이터(뒤집힌 문자)를 배제하는 것이 정밀한 위치 파악(Localization)에 실질적인 도움이 됨을 수치로 증명함.
*   **결정**: **"다양성(Exp 5) 이전에 무결성(Exp 4) 적용이 먼저임."** 
    - 만약 뒤집힌 각인이 방치된 채 Copy-Paste를 진행했다면, 모델은 '오염된 알약'을 더 많이 학습하며 혼란에 빠졌을 것임.
    - 똑바른 각인 정보가 복구되었으므로, 이 데이터를 기반으로 배경 누수를 제거하는 실험 5(Copy-Paste)를 하기로 결정

### [Exp 5] Copy-Paste (YOLO11s / 960px)
*   **분석**: 317장의 합성 데이터를 추가하여 데이터셋을 약 3배로 증량한 결과, **Kaggle Public Score 0.96804**라는 성적이 나옴. 
*   **통찰**: 로컬 mAP(0.99)는 누수 때문에 정체되었으나, 합성 데이터를 통한 '데이터 다양성 확보'가 실전 테스트 데이터(캐글)에서의 일반화 성능을 폭발적으로 향상시켰음을 입증함.
*   **결정**: 이제는 공간적 변형을 극한으로 주는 **실험 6(Rotation)**을 해보기로 결정.

### [Exp 6] Rotation Aug (YOLO11s / 960px)
*   **분석**: 전각도 회전(Degrees=180.0) 적용 결과, **Kaggle Public Score 0.85483**으로 대폭 하락. 
*   **통찰**: 캐글 테스트 데이터가 이미 정방향 위주로 정렬되어 있을 가능성이 높음. 과도한 회전 변형은 오히려 정방향 데이터에 대한 모델의 정밀도를 떨어뜨리는 독이 됨(Over-fitting to rotation-invariance). 
*   **결정**: 실험 7에서는 **회전을 다시 제거하고**, 성공했던 Exp 5 설정에 **1024px 해상도**만 추가

### [Exp 7] Final High-Res (YOLO11s / 1024px)
*   **분석**: 실험 5의 성공 베이스라인에 1024px 초고해상도를 투입한 결과, 로컬 mAP@50-95 **0.9934** 달성. 
*   **통찰**: 실험 6에서 배운 "테스트 셋 정방향 정렬" 특성을 적극 활용하여 회전을 제거하고, 해상도 증량(960px -> 1024px)에 집중함. 이는 알약의 미세 각인(OCR 특화) 및 테두리 정밀도를 극한으로 살리기 위한 전략임. 
*   **결정**: 해상도 단순 리사이즈는 임계점에 도달한 것으로 판단됨. YOLO11m으로 높이는 것은 보류하고, 기본기가 탄탄한 모델(Exp 5)을 기반으로 **1순위 전략(NMS 튜닝 및 Multi-scale 앙상블)** 을 하는 것이 맞다는 생각이 듬 
*   **재현**: `python train_yolov8.py --config configs/exp7_train.yaml`

### [Exp 8] Dynamic NMS Tuning (Validation Threshold Search)
*   **목적**: conf, iou 값을 바꿔가며 어떤 조합이 validation 성능(mAP@50-95)이 가장 좋은지 찾는 것
*   **방식**: 학습이 아니라 이미 학습된 best.pt로 val을 반복 평가
*   **분석**: Exp5 best 가중치 기준으로 `conf=0.20` 고정 후 `iou=0.50/0.60/0.70` 비교 시 mAP@50-95가 각각 **0.9926 / 0.9926 / 0.9925**로 거의 동일함.
*   **통찰**: 로컬 검증셋에서는 NMS 임계값 변화 영향이 매우 작고, `iou=0.70`은 미세하게 불리함.
*   **결정**: 실전 추론 기본값을 **`conf=0.20`, `iou=0.60`**으로 고정하고 다음 단계(Exp9, Exp10)로 진행.
*   **실험 결과 데이터**:
    | 구분 | IoU=0.50 | **IoU=0.60 (선택)** | IoU=0.70 |
    | :--- | :--- | :--- | :--- |
    | **mAP@50-95** | 0.9926 | **0.9926** | 0.9925 |
    | **mAP@50** | 0.9941 | **0.9941** | 0.9941 |

*   **재현(임계값 탐색)**: `python -u src/exp8_search.py --device 0 --verbose --confs 0.20 --ious 0.50,0.60,0.70`
*   **재현(추론)**: `python src/test_custom.py --model runs/pill_exp5_yolo11s_copypaste/weights/best.pt --imgsz 1024 --conf 0.20 --iou 0.60 --output submission/exp8_submission_iou0.6.csv`



---

## 🛠️ 실행 가이드 (How to Reproduce)

1.  **`preprocessing.py`** 상단 플래그 조정
    - `USE_STRATIFIED`: 계층적 분할 여부
    - `USE_COPY_PASTE`: Copy-Paste 증강 여부
2.  **`train_yolov8.py`** 실행
    - `--epochs`, `--imgsz`, `--name` 등의 인자 확인
    - 예시: `python train_yolov8.py --config configs/exp7_train.yaml`
3.  **`exp8_search.py`**로 NMS 임계값 탐색
    - 예시: `python -u src/exp8_search.py --device 0 --verbose --confs 0.20 --ious 0.50,0.60,0.70`
4.  **`test_custom.py`**로 제출 파일 생성
    - 예시: `python src/test_custom.py --model runs/pill_exp5_yolo11s_copypaste/weights/best.pt --imgsz 1024 --conf 0.20 --iou 0.60 --output submission/exp8_submission.csv`
