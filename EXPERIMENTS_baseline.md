# 🧪 Pill Detection: Experiment Logs

이 문서는 AI 9기 4팀의 모델 성능 향상 과정과 장애 해결 기록을 담고 있습니다.

## 📊 1. Baseline Experiment 
- **Date:** 2026-03-23
- **Objective:** 예원님의 정제된 라벨링 데이터를 활용한 고신뢰도 Baseline 구축.
- **Model:** YOLOv8 Nano
- **Params:** Epochs 50, Imgsz 640

### [Performance Metrics]
| Seed | Precision | Recall | mAP@50 | mAP@50-95 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.899 | 0.652 | 0.850 | 0.829 | |
| **123** | 0.879 | 0.691 | 0.898 | 0.869 | |
| **777** | 0.851 | 0.717 | 0.885 | 0.863 | |
| **AVG**| **0.876** | **0.686** | **0.877** | **0.853** | **성능 비약적 향상** |

> *참고:* mAP50 평균 **0.877** (이전 오염 데이터 Baseline 대비 약 2배 이상 성능 향상)

---

## 🔍 2. 기술적 장애 및 기회분석 
### 📌 장애 보고 
- **이슈:** `DATA_IMG_DIR` 경로 불일치로 이미지 로딩 실패.
- **원인:** 로컬 폴더 구조와 `.env` 파일의 상대 경로 설정 차이.
- **조치:** `data/raw/sprint_ai_project1_data/train_images`로 폴더 이동 및 `.env` 절대 경로/상대 경로 영점 조절 완료.

### 📌 성능 향상 기회 
- **라벨 정밀도:** 예원님의 수동 라벨링 json파일 사용 이후 mAP가 급상승함. **데이터의 질이 모델 크기보다 훨씬 중요함**을 실증적 지표로 확인.
- **Recall 분석:** 현재 평균 Recall(0.686)이 평균 Precision(0.876)보다 약 20% 낮음. 즉, 알약을 잘못 찾는(FP) 보다는 '아예 놓치는(FN)' 경우가 더 많음을 의미.
- **향후 해결책:**
  1. 가장 성적이 안 나오는 클래스를 파악하여 집중 분석 (클래스 불균형 해소 전략 연계).
  2. 아주 작은 알약에 대해 Augmentation 기법 (Rotation, Blur, Mosaic 조절) 추가 적용 검토.

---

## 📁 3. Artifacts Location
- 최적의 가중치 파일은 학습 완료 시 자동으로 최상단 `weights/` 경로로 추출됩니다.
- **Top Weights:** `/weights/baseline_best_seed_{seed}.pt`
- **Results Graphs & Logs:** `/runs/baseline/seed_{seed}/`
