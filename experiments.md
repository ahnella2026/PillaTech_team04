# Pill Detection Experiment Log

## 📑 지표 정의 및 측정 이유 (Metric Definitions & Rationale)

1.  **mAP@50 (Detection 성공률)**: "일단 찾기는 했는가?" 
    *   가장 기본적이고 너그러운 기준입니다. 알약의 존재를 놓치지는 않았는지를 판단하는 기초 실력입니다.
2.  **mAP@75 (Localization 정밀도)**: **"칼같이 찾았는가?"** 
    *   알약 테두리를 얼마나 정확하게 감쌌는지를 봅니다. 우리 프로젝트처럼 **알약의 각인(문자/로고)을 분석해야 하는 작업**이 뒤따른다면, 박스 오차가 작아야 글자가 잘리지 않고 정확한 OCR이 가능해집니다.
3.  **mAP@50-95 (종합 실력)**: "전체적인 품질이 어떤가?" 
    *   너그러운 기준(@50)부터 아주 깐깐한 기준(@95)까지의 평균입니다. 모델의 전반적인 완성도를 나타내는 가장 신뢰할 만한 점수입니다.
4.  **Best Epoch (최적화 지점)**: "언제 더 이상 학습할 필요가 없었는가?" 
    *   최대 성능을 낸 시점을 알아야 불필요한 과적합(Overfitting) 여부를 판단하고 학습 속도를 조절할 수 있습니다.


## 📊 실험 결과 요약 (Experiment Summary Table)

| ID | 실험명 | 분기 조건 (Split / Augment) | Model | Res | Epoch | **mAP50** | **mAP@50-95** | **mAP@75** | Best Epoch | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ref 1**| **Teammate (YOLOv8n)** | Random Split / Default Augs | v8n | 640 | 50 | **0.790** | 0.760 | - | - | 예원 레퍼런스 |
| **Exp 0** | **Baseline (YOLOv8n)** | Random Split / Default Augs | v8n | 640 | 50 | 0.959 | 0.935 | **0.950** | 50 | 극단적 데이터 누수 발생 |
| **Exp 1** | **Stratified Split** | **Stratified Split** / Default Augs | v8n | 640 | 20 | 0.402 | 0.384 | **0.395** | 20 | 계층적 분할 (20 epoch) |
| **Exp 1-2**| **Stratified (Full)** | Stratified Split / Default Augs | v8n | 640 | 50 | **0.960** | 0.943 | 0.945 | **36** | 분할만으로는 누수 완전 제어 불가 |
| **Exp 2** | **YOLO11 Pivot** | Stratified Split / Default Augs | **v11s** | 640 | 50 | **0.994** | **0.990** | **0.995** | **50** | 모델 교체 및 정밀도 수직 상승 (누수 지속) |
| **Exp 3** | **High-Res Pivot** | Stratified Split / Default Augs | **v11s** | 960 | 50 | **0.995** | **0.990** | **0.995** | **50** | 해상도 증가 불구, 누수 한계(Ceiling) 도달 확인 |
| **Exp 4** | **Integrated Opt** | Stratified + Copy-Paste | **v11s** | 960 | 50 | - | - | - | - | **최종 목표 (>0.90)** |

---

## 🔬 정성적 분석 및 이슈 사항 (Insights & Issues)

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

### [Exp 3] High-Res Pivot (YOLO11s / 960px)
*   **분석**: 이미지 해상도를 640px에서 960px로 대폭 키워 학습했음에도, mAP 지표는 기존 640px(Exp 2)와 거의 동일한 **한계점(Ceiling)**에 부딪힘.
*   **통찰**: 알약의 각인 등 세부 특징을 더 명확하게 제공(High-Res)하더라도, "기본적인 배경이 동일한 쌍둥이 이미지"라는 데이터셋 자체의 치명적 결함(Data Leakage)을 넘어서지는 못함.
*   **결정**: 960px의 고해상도 환경은 유지하되, 이 가짜 천장을 부숴버릴 실전형 데이터 파괴 증강 기법 **[Copy-Paste]** 의 즉시 도입이 요구됨.

---

## 🛠️ 실행 가이드 (How to Reproduce)

1.  **`preprocessing.py`** 상단 플래그 조정
    - `USE_STRATIFIED`: 계층적 분할 여부
    - `USE_COPY_PASTE`: Copy-Paste 증강 여부
2.  **`train_yolov8.py`** 실행
    - `--epochs`, `--imgsz`, `--name` 등의 인자 확인
