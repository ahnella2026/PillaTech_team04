# 💊PillaTech_team04
코드잇 스프린트 AI 9기 필라텍 4팀 1차 프로젝트

알약 식별 및 바운딩 박스 검출을 위한 객체 탐지(Object Detection) 프로젝트입니다.

---

## 🛠️ 협업 가이드 
4조 팀원분들은 `git pull`을 받은 후, 아래 순서대로 실행하여 학습 환경을 구축해 주세요.
### 1. 가상환경 및 라이브러리 설치
이 프로젝트는 **Python 3.12** 환경에 최적화되어 있습니다.

```bash
# 가상환경 생성 및 활성화
conda create -n codeit python=3.12 -y
conda activate codeit
```
# 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터셋 배치
원본 데이터 이미지는 용량 문제로 Git에 포함되지 않았습니다. **프로젝트 루트**에 아래 경로로 배치해 주세요.

- **이미지:** `./data/raw/sprint_ai_project1_data/train_images/`
- **어노테이션:** `./data/raw/sprint_ai_project1_data/train_annotations/`

### 3. 환경 변수 설정
루트의 `.env` 파일에 전처리 설정이 되어 있습니다. (기본 3-Seed 설정 완료)
수정이 필요한 경우 `.env` 파일의 `RANDOM_SEEDS` 등을 조정하세요.

### 4. 전처리 및 학습 자동화
```bash
# 1. 전처리 실행 (자동으로 .env 설정을 읽어옴)
python src/data/preprocessing.py

# 2. 학습 실행 (자동 3-Seed 학습 시작)
python train.py
```

💡 **Tip:** 학습이 완료되면 각 시드별 최고 성능 모델(`best.pt`)이 자동으로 최상위 `weights/` 폴더에 추출되어 저장됩니다
(※ 코드가 작동할 때 자동으로 다운로드되는 `yolov8n.pt`는 학습의 시작점이 되는 **사전 학습(Pre-trained) 베이스 모델**입니다.)
[상세 내역 보러가기 👉 EXPERIMENTS_baseline.md](./EXPERIMENTS_baseline.md)

---

## 5. 전처리 & 학습 전략 (Clean Baseline)
- **100% Data Inclusion:** 예원님의 수동 라벨링 작업을 통해 이전의 모든 오염 데이터(Invalid BBox 1건 포함)를 완벽히 정상화했습니다. 덕분에 단 한 장의 이미지 누락 없이 **전량 학습**에 반영되었습니다.
- **Label Restoration:** 8장의 누락 라벨 이미지를 복구하여 실제 물체 개수와 정답지(GT)를 완벽히 일치시켰습니다.
- **3-Seed 교차 검증:** 시드 42, 123, 777을 사용하여 데이터셋 분할 변수를 통제하고 공정한 성능(mAP 0.88)을 측정했습니다.
- 실험 로그 관리: 상세한 시드별 수치와 기록은 **[EXPERIMENTS_baseline.md](./EXPERIMENTS_baseline.md)**를 참고하세요.

## 6. 실험 결과 및 달성 현황 (Current Status)
1. **데이터 정상화 완료:** 예원님 수동 라벨링 전량 반영 (Invalid 0건, 누락 복구 완료).
2. **Clean Baseline 달성:** 3-Seed 평균 **mAP@50: 0.884** 돌파 (YOLOv8n 기준).

## 7. 향후 일정 (Next Steps)
1. **성능 고도화:** Nano 모델을 넘어 모델 사이즈 확장(Small, Medium) 및 Ensemble 기법 검토.
2. **증강 전략 최적화:** 알약 특성에 특화된 Rotation 및 Blur 증강 실험 진행 예정.
