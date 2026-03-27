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


## 📌 PillaTech Pill Detection Experiments Log

> [!IMPORTANT]
> **중대 기술 부채 발견 및 상환 (2026-03-26)**: 
> Exp 1~10까지의 실험 과정에서 아마도 **9건의 학습 데이터 오염**이 존재했을 것으로 추정됨을 10단계까지 한 후 알게 됨.  
> - **현상**: 8장의 라벨 누락(결측) 및 1건의 Invalid BBox(x=6567).
> - **영향**: 파이프라인의 `clip01()` 함수가 쓰레기 좌표를 화면 끝(`x_center=1.0`)에 강제 고정하여 모델에게 왜곡된 피처를 학습시킴.
> - **조치**: 
  1. 예원 브랜치(`pred/yewon`)로부터 `git restore` 명령어를 통해 원본 JSON 데이터 복구.
```
git switch model/exp-sujin
git restore --source=pred/yewon train_annotations
```
  2. `python preprocessing.py`를 실행하여 중간 산출물(`merged_annotations.json`)에 수정 사항 반영 (Report에서 `invalid bboxes: 0` 확인).
  3. `python prepare_yolo_dataset.py`를 실행하여 최종 YOLO TXT 레이블셋 최신화 및 `dataset.yaml` 업데이트.
> - **결과**: Exp 11부터는 **'Cleaned Baseline'**으로 명명하며 모든 지표의 신뢰성을 재확보함.

## 🧪 실험 목록 (Experiment Table)
| ID | 실험명 | 변경 사항 (Strategy) | Backbone | Size | Epoch | Seed | C | I | P | R | F1 | mAP@.5 | mAP@.5:.95 | Best | 인사이트 및 결과 |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Exp 1~10**| **Polluted-Sets** | **데이터 오염 상태에서의 실험들 (잠정 지표)** | - | - | - | - | - | - | - | - | - | - | 데이터 무결성 결함 잔존했을 것으로 추정됨 |
| **캐글** | **캐글용** | Stratified Split/욜로기본증강설정 | v8n | 640 | 50 | 50 | - | - | 0.79 | Kaggle: 0.70166 |
| **Ref 1**| **예원** | Random Split / 욜로기본증강설정 | v8n | 640 | 50 | **0.790** | 0.760 | - | - | 예원님 결과 |
| **Exp 0** | **YOLOv8n** | Random Split / 욜로기본증강설정 | v8n | 640 | 50 | 0.959 | 0.935 | **0.950** | 50 | • 극단적 데이터 누수 발생 <br>• 캐글 베이스라인 실험을 내 컴퓨터에서 실험한 결과 |
| **Exp 1-1** | **Stratified** | YOLO 기본증강 + Stratified | v8n | 640 | 20 | 0.402 | 0.384 | **0.395** | 20 | 계층적 분할 시작 |
| **Exp 1-2**| **Stratified** | YOLO 기본증강 + Stratified | v8n | 640 | 50 | **0.960** | 0.943 | 0.945 | **36** | 누수 잔존 확인 |
| **Exp 2** | **v11s Pivot** | YOLO 기본증강 + Stratified | **v11s** | 640 | 50 | **0.994** | **0.990** | **0.995** | **50** | 모델 교체 |
| **Exp 3** | **Hi-Res** | YOLO 기본증강 + Stratified | **v11s** | 960 | 50 | **0.995** | **0.990** | **0.995** | **50** | 해상도 증량 |
| **Exp 4** | **Flip-Off** | YOLO 기본증강 (Flip Off) | **v11s** | 960 | 50 | **0.995** | **0.992** | **0.995** | **50** | 각인 보호 효과 |
| **Exp 5** | **Custom CP** | YOLO 기본증강 (Flip Off) + 커스텀 CP | **v11s** | 960 | 50 | **0.995** | **0.993** | **0.995** | **50** | **Kaggle 0.968** (YOLO 내장 copy_paste옵션 아님) |
| **Exp 6** | **Rotation** | YOLO 기본 (Flip Off) + CP + Rotation | **v11s** | 960 | 50 | 0.990 | 0.940 | 0.995 | 50 | **Kaggle 0.854** |
| **Exp 7** | **Final-Res** | YOLO 기본 (Flip Off) + CP (No Rot) | **v11s** | 1024| 50 | **0.995** | **0.993** | **0.995** | **50** | 성능 정체 |
| **Exp 8** | **Dynamic NMS** | Exp5(best) 기준 NMS 튜닝 | v11s | 1024 | - | **0.9941** | **0.9926** | **0.9941** | - | Kaggle: **0.96670**, iou=0.70에서 0.0001 하락, 최종 conf=0.20, iou=0.60 고정 |
| **Exp 9** | **Multi-scale WBF** | Exp5 `best.pt` 기반 640/960/1024 추론 + WBF | v11s | Mix | - | **0.9932** | **0.9896** | **0.9932** | - | Kaggle: **0.97243** (최고점),Best Epoch는 Exp 9가 추론-only 앙상블이라 그대로 - |
| **Exp 10-1**| **Seed 42** | Exp 5 (Seed 42) | v11s | 960 | 50 | 0.9949 | 0.9942 | 0.9950 | 50 | 개별 시드 훈련 1 |
| **Exp 10-2**| **Seed 123** | Exp 5 (Seed 123) | v11s | 960 | 50 | 0.9950 | 0.9904 | 0.9950 | 50 | 개별 시드 훈련 2 |
| **Exp 10-3**| **Seed 777** | Exp 5 (Seed 777) | v11s | 960 | 50 | 0.9944 | 0.9932 | 0.9950 | 50 | 개별 시드 훈련 3 |
| **Exp 10(F)**| **3-Seed Ens**| Exp 10-1~3 WBF 앙상블 | v11s | 960 | - | **0.9942** | **0.9927** | **0.9942** | - | **Kaggle: 0.98073 (현재 최고점!!)** 앙상블의 일반화 성능 증명 |
| **Exp 11** | **Dirty-Aug** | Exp 5 Clean (fliplr: 0.5 오설정) | v11s | 960 | 50 | 0 | .25 | .70 | 0.9696 | 0.9696 | 0.9696 | 0.9931 | 0.9899 | 50 | **[기각]** 변인 통제 실패 (Kaggle: 0.95910) |
| **Exp 12** | **Cleaned-Base** | **Exp 5 Clean (fliplr: 0.0 복구)** | v11s | 960 | 50 | 0 | .25 | .70 | **0.9740** | **0.9649** | **0.9694** | **0.9946** | **0.9933** | 50 | **[신뢰 베이스라인 세팅 완료]** 데이터 정제 ROI 확인 (Kaggle: 0.96528) |
| **Exp 13** | **DETR Base** | Transformer 기반 모델 | DETR | 1024 | 50 | - | - | - | - | 진행 예정 |
| **Exp 14** | **Hetero WBF** | v11s + v26n + DETR 융합 | Mix | Mix | - | - | - | - | - | 진행 예정 |
| **Exp 15** | **Pseudo-L** | Test 셋 Pseudo-label 재학습 | Mix | Mix | - | - | - | - | - | 진행 예정 |
| **Exp 16** | **Colab Pro** | v11m, v12-S/M 고체급 학습 | Mix | 1024 | 50 | - | - | - | - | 진행 예정 |


### 평가 기준 정리
*   Exp 1~9의 mAP 지표는 모두 로컬 `validation` 셋(`data/yolo_dataset/images/val`) 기준으로 산출함.
*   캐글 제출용 CSV는 별도 추론 스크립트 `src/test_custom_v12.py`를 사용해 `test_images` 기준으로 생성함.
*   따라서 로컬 mAP와 캐글 점수는 서로 다른 데이터셋에서 측정된 값이며, 직접적으로 동일 지표가 아님.
*   Exp 8은 `validation` 기준으로 Exp5의 `best.pt`를 사용해 NMS 탐색을 수행한 뒤, 동일 가중치로 `test_images` 제출 CSV를 생성한 실험임.

### 일반 실험과 앙상블 실험의 차이
*   일반 단일 모델 실험(Exp 1~8)은 `validation` 셋에 대해 직접 평가하여 mAP를 산출하며, 별도의 `val` 예측 CSV 저장은 필수가 아님.
*   앙상블 실험(예: Exp 9)은 여러 예측 결과를 먼저 병합한 뒤 최종 예측으로 mAP를 계산해야 하므로, 검증 과정에서 해상도별 또는 모델별 중간 예측이 일시적으로 필요할 수 있음.
*   다만 `val` 기준 중간 예측은 로컬 검증용 임시 산출물일 뿐이며, 캐글 제출용 파일은 항상 `test_images` 기준 최종 CSV만 사용해야 0점처리가 안됨(주의!)
*   향후 앙상블 실험에서도 원칙은 동일하며, `val` 중간 산출물은 필요 시에만 생성하고 최종 기록에는 mAP 수치만 남길 것임


---
## 🔬 정성적 분석 및 이슈 사항 (Insights & Issues)
### [우선순위 기반] 성능 최적화 후속 전략 (Post-Exp8 Strategy)
Exp 7(1024px 리사이즈) 실험 이후, 훈련 비용이 큰 모델 업그레이드는 보류하고 **캐글 실전 제출 횟수(하루 2회)를 효율적으로 사용하기 위한 단기/고도화 기법**을 최우선적으로 사용하기로 함. 

1.  **🔥 0순위 (데이터 무결성 검증 및 ROI 측정)** 
    *   **Cleaned Baseline (Exp 11)**: 기존 Exp 5(가장 성능이 좋았던 모델) 설정을 유지하되, 복구된 깨끗한 데이터셋(8장 결측 보정 + 1건 Invalid BBox 수정)으로 재학습하여 **데이터 정제만으로 발생하는 순수 성능 향상분**을 측정. 
    *   이를 통해 오염된 데이터가 mAP에 미치는 실질적 악영향을 수치화하고, 새로운 신뢰 베이스라인으로 확립함.
2.  **🔥 1순위 (최신 아키텍처 확장 - 고성능 모델 도입)** 
    *   **YOLO26-n & RT-DETR (Exp 12, 13)**: 데이터 무결성이 확보된 상태에서, 2026년형 NMS-free 모델과 Transformer 기반 모델을 동원해 mAP 1.0에 도전. 
    *   **NMS 튜닝 (Exp 8, 완료)**: 기존 분기에서 최적화 지표 확보.
3.  **🔵 3순위 (최종 병기 - 앙상블 및 고급기법)**
    *   **Hetero WBF (Exp 14)**: 각기 다른 장점을 가진 YOLO와 RT-DETR의 예측값을 WBF로 융합하여 mAP 극대화.
    *   **Pseudo-Labeling (Exp 15)**: 가장 강력하게 앙상블된 결과를 Test셋의 레이블로 삼아 추가 재학습 진행.
4.  **💎 번외 (Colab Pro 유료 환경 활용 - A100 GPU 권장)**
    *   **고체급 모델 부스팅 (Exp 16)**: 현재 로컬(RTX 3060 12GB) 환경에서는 OOM(Out of Memory)으로 학습 불가능한 `YOLO11m` 모델이나, 계산량이 많아 CPU/저가형 GPU에서 느린 **2025년 최신 `YOLOv12-S/M` (Attention 기반 고성능 모델)**을 학습시킵니다. Colab Pro의 고용량 GPU(A100/V100)를 대여하면 1024px 이상의 고해상도 환경에서 높은 Batch-size로 압도적인 성능 점프가 가능할 것으로 기대하고 있음 


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

### [Exp 5] Custom Copy-Paste Synthesis (YOLO11s / 960px)
*   **분석**: 317장의 합성 데이터를 추가하여 데이터셋을 약 3배로 증량한 결과, **Kaggle Public Score 0.96804**라는 성적이 나옴. 
*   **통찰**: 로컬 mAP(0.99)는 누수 때문에 정체되었으나, 커스텀 합성 데이터를 통한 '데이터 다양성 확보'가 실전 테스트 데이터(캐글)에서의 일반화 성능을 폭발적으로 향상시켰음을 입증함. (주의: YOLO 내부 하이퍼파라미터 `copy_paste`를 켠 것이 아님)
*   **결정**: 이제는 공간적 변형을 극한으로 주는 **실험 6(Rotation)**을 해보기로 결정.

### [Exp 6] Rotation Aug (YOLO11s / 960px)
*   **분석**: 전각도 회전(Degrees=180.0) 적용 결과, **Kaggle Public Score 0.85483**으로 대폭 하락. 
*   **통찰**: 캐글 테스트 데이터가 이미 정방향 위주로 정렬되어 있을 가능성이 높음. 과도한 회전 변형은 오히려 정방향 데이터에 대한 모델의 정밀도를 떨어뜨리는 독이 됨(Over-fitting to rotation-invariance). 
*   **결정**: 실험 7에서는 **회전을 다시 제거하고**, 성공했던 Exp 5 설정에 **1024px 해상도**만 추가

### [Exp 7] Final High-Res (YOLO11s / 1024px)
*   **분석**: 실험 5의 성공 베이스라인에 1024px 초고해상도를 투입한 결과, 로컬 mAP@50-95 **0.9934** 달성. 
*   **통찰**: 실험 6에서 배운 "테스트 셋 정방향 정렬" 특성을 적극 활용하여 회전을 제거하고, 해상도 증량(960px -> 1024px)에 집중함. 이는 알약의 미세 각인(세부 특징 추출) 및 테두리 정밀도를 극한으로 살리기 위한 전략임. 
*   **결정**: 해상도 단순 리사이즈는 임계점에 도달한 것으로 판단됨. YOLO11m으로 높이는 것은 보류하고, 기본기가 탄탄한 모델(Exp 5)을 기반으로 **1순위 전략(NMS 튜닝 및 Multi-scale 앙상블)** 을 하는 것이 맞다는 생각이 듬 
*   **재현**: `python train_yolov8.py --config configs/train/exp7_train.yaml`

### [Exp 8] Dynamic NMS Tuning (Validation Threshold Search)
*   **목적**: conf, iou 값을 바꿔가며 어떤 조합이 validation 성능(mAP@50-95)이 가장 좋은지 찾는 것
*   **방식**: 학습이 아니라 Exp5의 `best.pt`를 사용해 `validation` 셋에서 반복 평가한 뒤, 최종 설정으로 `test_images` 제출 CSV를 생성함.
*   **분석**: Exp5 best 가중치 기준으로 `conf=0.20` 고정 후 `iou=0.50/0.60/0.70` 비교 시 mAP@50-95가 각각 **0.9926 / 0.9926 / 0.9925**로 거의 동일함.
*   **통찰**: 로컬 검증셋에서는 NMS 임계값 변화 영향이 매우 작고, `iou=0.70`은 미세하게 불리함.
*   **결정**: 실전 추론 기본값을 **`conf=0.20`, `iou=0.60`**으로 고정하고 다음 단계(Exp9, Exp10)로 진행.

*   **실험 결과 데이터**:
    | 구분 | IoU=0.50 | **IoU=0.60 (선택)** | IoU=0.70 |
    | :--- | :--- | :--- | :--- |
    | **mAP@50-95** | 0.9926 | **0.9926** | 0.9925 |
    | **mAP@50** | 0.9941 | **0.9941** | 0.9941 |

*   **재현(임계값 탐색)**: `python -u src/exp8_search.py --device 0 --verbose --confs 0.20 --ious 0.50,0.60,0.70`
*   **재현(추론)**: `python src/test_custom_v12.py --model runs/pill_exp5_yolo11s_copypaste/weights/best.pt --imgsz 1024 --conf 0.20 --iou 0.60 --output submission/exp8_submission_iou0.6.csv`



### [Exp 9] Multi-Scale WBF (다중 해상도 앙상블)
*   **배경**: 원래 계획했던 Snapshot 앙상블은 과거 epoch 가중치가 남아 있지 않아 진행 불가. 따라서 추가 학습 없이도 앙상블 효과를 노릴 수 있는 `Multi-scale WBF`로 전략을 전환함.
*   **방식**: Exp5의 최종 가중치 `runs/pill_exp5_yolo11s_copypaste/weights/best.pt` 하나를 사용해 `640 / 960 / 1024` 세 해상도로 각각 추론한 뒤, 생성된 CSV를 WBF로 병합함.
*   **실행 결과**: 개별 추론 결과 3개(`exp9_submission_640.csv`, `exp9_submission_960.csv`, `exp9_submission_1024.csv`)를 생성했고, WBF 적용 후 최종 제출 파일 `submission/exp9_final_wbf.csv`를 생성함. 이후 `validation` 기준 재평가를 별도로 수행하여 로컬 mAP 지표를 기록함.
*   **로컬 검증 결과**: `metrics/exp9_val_metrics.json` 기준 `mAP50=0.9932`, `mAP@50-95=0.9896`, `mAP@75=0.9932`.
*   **통찰**: 재학습 없이 단일 가중치의 다중 해상도 예측을 WBF로 병합하는 방식만으로도, 로컬 `validation` 기준 높은 성능을 유지하면서 Kaggle Public Score **0.97243**까지 도달함.
*   **시행착오**: Exp 9는 처음에는 `test_images` 기반 제출용 WBF 실험으로만 수행되었고, 이후 표 기록 및 근거 정리를 위해 `validation` 기준 WBF 평가를 별도로 추가 수행함.
*   **재현**: `bash scripts/exp9/run_exp9_val.sh` (로컬 validation 평가), `bash run_exp9.sh` (캐글 제출용 test 추론)

### [심층 리포트] NMS vs WBF의 IoU 파라미터 역할 차이 (Exp 8 vs Exp 9)
Exp 8에서 로컬 검증 기준 최적 밸런스(`mAP@50-95=0.9926`)를 보인 `iou=0.60` 설정을 Exp 9의 멀티스케일 앙상블 파이프라인에 적용함. 다만 Exp 9의 Kaggle 0.97243은 `iou=0.60` 하나의 효과라기보다, **멀티스케일 추론 + 해상도별 NMS + WBF 병합이 함께 작동한 결과**로 해석하는 것이 더 정확함. 

**1. 해상도별 추론 단계의 NMS (`test_custom_v12.py`)**
*   단일 모델 NMS에서 `iou=0.60`은 기본값 `0.70`보다 박스 억제 조건이 더 공격적임. 즉, 조금만 겹쳐도 중복 박스로 판단해 제거할 가능성이 더 높음.
*   Exp 8에서는 이 설정이 단일 모델 추론 환경에서 Kaggle 점수를 소폭 하락시킴. 따라서 단일 모델 기준으로는 `0.60`이 항상 유리하다고 볼 수는 없음.
*   하지만 Exp 9처럼 640/960/1024 세 해상도에서 예측이 동시에 생성되는 환경에서는, 각 해상도 단계에서 중복 박스를 1차로 정리하는 역할을 했다고 볼 수 있음. 

**2. 앙상블 단계의 WBF (`ensemble_wbf.py`)**
*   WBF의 `iou=0.60`은 병합 기준으로 작동함. 따라서 `0.70`보다 더 쉽게 같은 객체로 묶이게 만드는 조건임.
*   해상도별로 조금씩 어긋난 박스들이 너무 엄격한 기준 때문에 따로 남지 않도록, `0.60` 기준으로 더 유연하게 병합할 수 있었음. 

**[결론]**
Exp 8에서 단일 모델 기준으로는 다소 불리했던 `iou=0.60` 설정이, Exp 9의 멀티스케일 WBF 파이프라인에서는 충분히 잘 작동했음. 다만 Exp 9의 성능 향상은 `0.60` 하나 때문이라기보다, **다중 해상도 예측과 WBF 병합 구조 전체의 효과**로 보는 것이 적절함.

### [Exp 10] 3-Seed Ensemble Training (The Masterpiece)
*   **배경**: 단일 모델(Exp 5)의 한계를 극복하기 위해 3-Seed(42, 123, 777) 기반의 모델 분산(Variance) 제어 전략 도입.
*   **실험 결과**:
    | ID | Seed | mAP50 | mAP75 | mAP50-95 |
    | :--- | :--- | :--- | :--- | :--- |
    | Exp 10-1 | 42 | 0.9949 | 0.9950 | **0.9942** |
    | Exp 10-2 | 123 | 0.9950 | 0.9950 | **0.9904** |
    | Exp 10-3 | 777 | 0.9944 | 0.9950 | **0.9932** |
*   **결과**: **Kaggle Public Score 0.98073**
*   **통찰**: 
    1.  **앙상블의 압도적 위력**: 로컬 mAP(0.9927)는 누수 때문에 단일 모델과 비슷해 보였으나, 캐글 리더보드에서는 **+0.0083**이라는 점수 상승을 보여줌.
    2.  **데이터 오염의 역설**: 비록 데이터에 일부 결함이 있었음에도 불구하고, 시드 다양성이 그 결함으로 인한 왜곡을 '상쇄'하며 강력한 강건성(Robustness)을 보여줬음.
    3. 3개 시드의 평균 mAP50-95는 **0.9926 (±0.0019)** 로 매우 높은 안정성을 보였고, 시드 42가 개별 최고 성능(0.9942)을 기록함. 실무에서는 3개 시드를 통해 검증하는 작업이 중요하지만 한정된 시간 안에 캐글 점수 올리는 데는 불필요한 작업일 수도 있었으나 성적이 올라서 원인파악을 해야 함. 

*   **최종 앙상블 결과**: 3개 시드의 validation 예측을 WBF로 병합한 결과, `metrics/exp10_final_ensemble_metrics.json` 기준 **mAP50=0.9942 / mAP75=0.9942 / mAP@50-95=0.9927** 을 기록함.
*   **조치**: `submission/exp10_final_submission.csv`를 현재까지의 마스터피스로 선언. 향후 모든 고도화 모델은 이 점수를 넘어설 것을 목표로 함.

### [Exp 11] Dirty-Aug Fail (Lesson Learned)
*   **원인**: 데이터는 정제했으나 `fliplr: 0.5` 옵션이 켜져 Exp 5와 1:1 비교 불가 상태로 진행됨.
*   **교훈**: 정제 후 성능 하락 시, 데이터 자체가 아닌 '실험 환경 매칭' 유무를 먼저 파악해야 함. 
*   **현상**: Exp 11 제출 결과 Kaggle 점수가 `0.9680` -> `0.9591`로 하락. 
*   **원인**: 데이터뿐만 아니라 `fliplr`(수평 뒤집기) 옵션이 Exp 5(`0.0`)와 달리 **`0.5`**(기본값)로 잘못 적용됨. 
*   **통찰**: 수평 뒤집기가 활성화되면서 알약 내 텍스트 정보(각인) 특징이 오염되어, 데이터 정제 효과가 가려짐.
*   **조치**: No-Flip(`fliplr: 0.0`) 환경을 완벽히 복원한 **Exp 12**를 통해 데이터 정제 ROI를 재측정함. 
*   **교훈**: A/B 테스트 시 오직 단 하나의 변수(데이터)만 변경해야 하며, 다른 모델 하이퍼파라미터는 100% 일치시켜야 함.
*   **재현**: `python train_yolov11.py --config configs/train/exp11_train_yolo11s_flip.yaml`

### [Exp 12] Cleaned Baseline (The True Outcome)
*   **현상**: Exp 11의 변인 통제 오류(fliplr: 0.5)를 바로잡고, No-Flip(`0.0`) 환경에서 재학습 완료.
*   **결과**: **mAP@50 0.9946** (+0.0015), **mAP@50-95 0.9933** (+0.0034) 달성 (Exp 11 대비).
*   **통찰**: 
    1.  **BBox 정밀도 대폭 향상**: mAP@50-95 지표가 0.99대로 진입하며 우리가 정제한 '깨끗한 좌표'가 물리적 정밀도를 극대화했음을 증명함.
    2.  **노이즈 제거의 승리**: 뒤집힌 텍스트 특징을 배제함으로써 Precision(`0.9740`)이 개선됨.
*   **조치**: `runs/exp12_train_yolo11s_noflip/weights/best.pt`를 최종 Cleaned Baseline으로 확정.
*   **추론**: `python src/test_custom_v12.py --config configs/inference/exp12_inference_yolo11s_noflip.yaml`
*   **재현**: `python train_yolov11.py --config configs/train/exp12_train_yolo11s_noflip.yaml`




---

## 💾 [L3 실무형] 모델 체크포인트(Weights) 운영 정책
추후 뼈아픈 가중치 유실 사태(Snapshot 기회 낭비)를 방지하기 위해 다음과 같은 엄격한 저장 모델 운영 규칙을 신설 및 적용

**1. 모든 [학습 실험]은 최소 4가지를 남긴다**
*   `best.pt`, `last.pt` 
*   학습 설정 파일 (`configs/train/expN_train.yaml`)
*   실험 기록 (`experiments.md`에 실행 커맨드 + 환경 + 핵심 결과)

**2. [추론-only 실험]은 체크포인트를 만들지 않는다**
*   **적용**: Exp 9 (Multi-scale WBF), Exp 13 (Hetero WBF)
*   **저장물**: `configs/inference/expN_inference.yaml` 설정 파일, 제출망 `submission.csv`, 모듈별 독립 분석 예측 로그(필요시)

**3. 라이트급 모델(yolo11s, v26n 등)은 무조건 `save_period=1` 강제**
*   **적용**: Exp 10 (3-Seed), Exp 11 (v26n), Exp 12 (RT-DETR), Exp 14 (Pseudo-Labeling)
*   **이유**: 가중치 하나당 최고 19MB 수준으로 로컬(RTX 3060) 환경에서 50개(1GB)를 모두 저장해도 I/O 병목 및 용량 압박이 전무함. 차후 **최근 3개 에폭**을 이용한 Snapshot 앙상블을 100% 보장하기 위함.

**4. 고체급 파운데이션 모델(yolo11m, v12-S 이상)은 Queue(최근 3개) 콜백 적용**
*   **적용**: Exp 15 (Colab 고체급 모델 학습 등)
*   **원칙**: 초거대 모델은 1에폭만 저장해도 수 기가의 비용이 발생하므로 절대 `save_period=1`을 남발하지 않음. Custom Callback 코드를 개조하여 훈련 중 **[저장 시 가장 오래된 과거 가중치 1개 실시간 폐기(os.remove)]** 하는 링 버퍼 형식으로 용량 상한 고정.

**5. 수동 가비지 컬렉션 (G.C.)**
*   최종 성능을 평가한 뒤 "더 이상 앙상블(Snapshot) 요소가 없다"고 결론 나면 `best.pt`, `last.pt`만 남기고 과감하게 중간 에폭 파일들은 삭제하여 디스크를 최적화한다. 
*   **[업데이트 예정]**: 학습 루틴 종료 시 터미널에 **'스토리지 최적화 가이드'** 메시지를 자동으로 출력한다. (예: "Snapshot 앙상블 추천 에폭: [42, 48, 50], 나머지 삭제 권장")

**6. 명명 규칙 (Tri-Match Naming Policy)**
*   **원칙**: `학습파일명` == `YAML 내 name` == `추론파일명` == `runs 폴더명`을 1:1:1:1로 일치시킨다.
*   **예시**: 
    *   학습: `configs/train/exp12_train_yolo11s_noflip.yaml`
    *   내부: `name: "exp12_train_yolo11s_noflip"` (결과물은 `runs/exp12_train_yolo11s_noflip` 자동 생성)
    *   추론: `configs/inference/exp12_inference_yolo11s_noflip.yaml`
*   **이유**: 수백 개의 실험이 쌓였을 때, 특정 가중치가 어떤 설정(해상도, 모델, 증강)으로 학습되었는지 즉각적으로 추적하기 위함.

---

## 🛠️ 실행 가이드 (How to Reproduce)

1.  **`preprocessing.py`** 설정 
    * `USE_STRATIFIED`: 계층적 분할 여부 세팅
    * `USE_COPY_PASTE`: Copy-Paste 증강 여부 세팅
2.  **`train_yolov11.py`** 실행 (학습)
    * `python train_yolov11.py --config configs/train/exp11_train_yolo11s_flip.yaml`
3.  **`exp8_search.py`** 실행 (NMS 최적화)
    * `python -u src/exp8_search.py --device 0 --verbose --confs 0.20 --ious 0.50,0.60,0.70`
4.  **`test_custom_v12.py`** 실행 (추론 + Config 자동 반영)
    * `python src/test_custom_v12.py --config configs/inference/exp8_inference.yaml`
