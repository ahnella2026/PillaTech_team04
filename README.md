# PillaTech_team04
코드잇 스프린트 AI 9기 4팀 1차 프로젝트

# 🧪 PillaTech 희귀 클래스 데이터 증강 실험 (Augmentation Study)

본 실험은 데이터 불균형 문제를 해결하기 위해 팀원들이 각자 구현한 세 가지 증강 기법의 성능을 비교 분석하기 위한 용도입니다. 모든 코드는 `--exp_name` 인자를 통해 구분되어 실행됩니다.

---

## 📊 실험 기법 및 담당자 (Method Comparison)

| 기법 (Method) | 담당자 | `exp_name` 설정값 | 설명 |
| :--- | :---: | :---: | :--- |
| **Oversampling** | 예원 | `oversampling` | 희귀 클래스 이미지를 단순 복제하여 비중을 높임 |
| **Copy-Paste V1** | 수진 | `copy_paste_v1` | 희귀 알약 객체를 크롭하여 다른 이미지에 합성 (배경 고정형) |
| **Copy-Paste V2** | 한별 | `copy_paste_v2` | 객체 회전 및 랜덤 스케일링이 추가된 고도화된 합성 기법 |

---

## 🚀 실행 가이드 (How to Run)

원하는 기법의 `[EXP_NAME]`을 선택하여 아래 순서대로 터미널에 입력하세요.  
(예: 한별 기법 테스트 시 `copy_paste_v2` 입력)

### 1️⃣ 데이터 전처리 및 증강 생성
```bash
python preprocessing_aug.py --exp_name [EXP_NAME]
```

### 2️⃣ YOLO 데이터셋 빌드
생성된 증강 이미지와 원본 이미지를 합쳐 YOLO 학습용 폴더 구조를 생성합니다.
```bash
python prepare_yolo_dataset_aug.py --exp_name [EXP_NAME]
```

### 3️⃣ 모델 학습 및 평가
```bash
python train_yolo_aug.py --exp_name [EXP_NAME]
```

## 📂 주요 파일 설명 (Scripts)
- preprocessing_aug.py: 클래스별 빈도 분석 및 각 기법(Copy-Paste, Oversampling) 함수가 통합된 전처리 스크립트입니다.

- prepare_yolo_dataset_aug.py: 오프라인 증강된 이미지를 포함하여 실제 학습 가능한 YOLO 포맷으로 변환합니다.

- train_yolo_aug.py: 실험별 하이퍼파라미터를 관리하며 학습 종료 후 metrics/ 폴더에 결과를 저장합니다.