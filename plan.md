# [Exp 19~22+] YOLO11s Online Augmentation Optimization Plan

> **공식 트랙(Team Official)**: Exp 15 (`960`) / Kaggle: **0.96455**  
> **확장 트랙(Extension)**: Exp 17 (`1024`) / Kaggle: **0.97292**  
> **참고 실험**: Exp 18 (`1280`)은 해상도 벤치마크용(시간 과다로 온라인 증강 풀탐색 대상 아님)  
> **Target**: 단일 모델 Kaggle ≥ 0.970 유지/개선 → 3-Seed Ensemble ≥ 0.985

> [!IMPORTANT]
> **운영 원칙 (제출 안정성 우선)**
> - 의사결정/제출은 **팀 공통 960 트랙 결과**를 기준으로 한다.
> - `1024`는 **비차단 확장 트랙**으로 운영한다(시간 여유 시 top 후보만 재검증).
> - 실험 기록은 반드시 분리한다: `Team Official (960)` vs `Extension (1024)`.

---

## 0. 📎 AI 리뷰 종합 (Gemini / Claude / Codex)

### 3개 AI의 공통 합의사항
1. **`fliplr=0.0`, `flipud=0.0`, `mixup=0.0` 강제 유지** — 실험 데이터(Exp 4, 6, 11, 12)에서 이미 검증 완료.
2. **Seed를 42로 고정**하고 파라미터 튜닝 → 최종 확정 후에만 Multi-Seed(42/123/777) 앙상블.
3. 온라인 증강 튜닝의 기대치는 **+0.002~0.006 수준**이며, "점수를 큰 폭으로 올리는 것"보다 **"점수를 깎는 독성 설정 제거"**가 핵심 가치.

### AI별 핵심 차이 비교

| 항목 | Gemini | Claude | Codex |
|:---|:---|:---|:---|
| `hsv_h=0.015` | "치명적, 0.005 이하 또는 OFF" | "과장. ±1.5%는 안전. 실험에서 OFF 비교" | "과장에 동의. 단일변수 실험 필요" |
| `degrees` | `15.0` 적극 권장 | **`0.0` 유지 권장** (이 도메인에 불필요) | "후순위. 도메인에서 이득 불확실" |
| CLAHE | 최우선 추가 권장 | "현재 bottleneck과 무관. 후순위" | 언급 없음 |
| 실험 수 | 4개 (Exp 16~19) | 2~3개로 압축 | 1차 단일변수 4개만 |
| 실험 방법론 | 파라미터 그룹 추가/변경 | 제거부터 시작 (Ablation-first) | **단일변수 통제**(1차) → 조합(2차) |
| `auto_augment` | Albumentations로 교체 | 먼저 꺼보고 효과 측정 | `none`으로 끄고 비교 |
| YOLO `copy_paste` | 언급 없음 | 언급 없음 | 언급 없음 |

### Claude의 최종 판단

> **Codex 의견이 가장 실전적이다.** 이유: Exp 16에서 `hsv_h + erasing + auto_augment`를 한 번에 끄면 점수가 올랐을 때 "어떤 놈이 독이었는지" 특정할 수 없다. Codex의 "단일변수 4개 → 상위 2개 조합"이 원인분리(Ablation) 원칙에 가장 충실하다.

> **보완 메모(2026-03-31)**: YOLO 내장 `copy_paste`는 핵심 후보로 검토했으나, 현재 데이터셋 라벨 포맷(전량 bbox 5컬럼)에서는 실질적으로 no-op라 보류한다. 상세는 Section 5 참조.

---

## 1. 🛑 현재 상태 진단 (Exp 15 기준)

### Exp 15 증강 설정 현황 vs YOLO 디폴트

> [!WARNING]
> **[소스코드 검증 완료 — 2026-03-31]** Ultralytics 8.4.23 기준, `v8_transforms()` (detect 파이프라인)에서 실제로 사용(호출)되는 파라미터만 ✅로 표기. `auto_augment`와 `erasing`은 `classify_augmentations()`에만 존재하며 **detect 학습에서는 완전 no-op(효과 0)**임이 소스코드 레벨에서 확인됨. (Codex 분석 인용)
> - detect 경로: `augment.py` L.2384 (`v8_transforms`)
> - classify 전용 경로: `augment.py` L.2509 (`classify_augmentations`)

| 파라미터 | Exp 15 값 | YOLO 디폴트 | detect 파이프라인 작동 여부 | 독성 의심 |
|:---|:---:|:---:|:---|:---:|
| `hsv_h` | 0.015 | 0.015 | ✅ `RandomHSV`에서 작동 | ❓ 검증 대상 |
| `hsv_s` | 0.7 | 0.7 | ✅ `RandomHSV`에서 작동 | ✅ 안전 |
| `hsv_v` | 0.4 | 0.4 | ✅ `RandomHSV`에서 작동 | ✅ 안전 |
| `translate` | 0.1 | 0.1 | ✅ `RandomPerspective`에서 작동 | ❓ 검증 대상 |
| `scale` | 0.5 | 0.5 | ✅ `RandomPerspective`에서 작동 | ❓ 검증 대상 |
| `shear` | 0.0 | 0.0 | ✅ `RandomPerspective`에서 작동 | ✅ 안전 (0.0 고정) |
| `perspective` | 0.0 | 0.0 | ✅ `RandomPerspective`에서 작동 | ✅ 안전 (0.0 고정) |
| `fliplr` | **0.0** | ~~0.5~~ | ✅ `RandomFlip`에서 작동 | ✅ 해결됨 |
| `flipud` | 0.0 | 0.0 | ✅ `RandomFlip`에서 작동 | ✅ 안전 |
| `degrees` | 0.0 | 0.0 | ✅ `RandomPerspective`에서 작동 | ✅ 안전 (후순위 검토) |
| `mosaic` | 1.0 | 1.0 | ✅ `Mosaic`에서 작동 | ❓ 검증 대상 |
| `copy_paste` | 0.0 | 0.0 | ⚠️ 작동하나 **세그멘트 마스크 없으면 즉시 스킵** (`augment.py` L.1713) | 🔵 Step2.5 게이트 |
| `mixup` | 0.0 | 0.0 | ✅ `MixUp`에서 작동 | ✅ 안전 (0.0 고정) |
| `erasing` | 0.4 | 0.4 | ❌ **detect에서 no-op** (classify 전용) | ~~❓~~ → **가짜 파라미터** |
| `auto_augment` | randaugment | randaugment | ❌ **detect에서 no-op** (classify 전용) | ~~❓~~ → **가짜 파라미터** |

**실제 독성 의심 파라미터 (detect에서 작동하는 것 한정)**: `hsv_h`, `mosaic`, `translate`, `scale`

---

## 2. 📋 증강 기법 그룹화 (도메인 기반)

| 그룹 | 해당 항목 | 전략 | 이유 |
|:---|:---|:---|:---|
| **🚫 확정 금지** | `fliplr`, `flipud`, `mixup`, `degrees(>15)` | **0.0 고정** | Exp 4/6/11/12에서 실증. 각인 파괴 및 정방향 데이터 불일치. |
| **❓ 독성 검증 필요** | `hsv_h`, `mosaic`, `translate`, `scale` | **단일변수 통제 실험** | detect 파이프라인에서 실제 작동하며, 현재 도메인 적합성이 미검증. |
| **🟢 안전 유지** | `hsv_s`, `hsv_v` | **현행 유지** | 조명/채도 변동 모사에 유효했고, 현재까지 뚜렷한 독성 신호 없음. |
| **⏸️ 보류** | `copy_paste` (YOLO 내장), `auto_augment`, `erasing` | **실험 제외** | `copy_paste`: 세그멘트 마스크 부재 시 no-op. `auto_augment/erasing`: detect에서 no-op. |

---

## 3. 🔬 실험 실행 계획 (단일변수 통제 + 2단계 해상도 전략)

> [!CAUTION]
> **번호 규칙 재정의**: `Exp18`은 **1280 해상도 벤치마크**로 고정한다.  
> 온라인 증강 실험은 **`Exp19`부터 시작**한다.

> [!IMPORTANT]
> **해상도 운영 전략 (Team 960 + Extension 1024)**
> 1. **공식 경로**: `960`에서 단일변수/조합 실험 수행(팀과 동일 조건)  
> 2. **확장 경로**: 공식 경로 top 후보를 `1024`에서 비차단 재검증

> [!IMPORTANT]
> **변수 통제 원칙**
> - 모든 실험은 **Seed 42** 고정
> - `fliplr=0, flipud=0, mixup=0, copy_paste=0, shear=0, perspective=0, degrees=0` 유지
> - Step 1에서는 **한 번에 1개 변수만** 변경

### Phase 0: 해상도 벤치마크 마감

| 실험 ID | 목적 | 상태 |
|:---:|:---|:---|
| **Exp 18** | `imgsz=1280` 해상도 성능/시간 벤치마크 | 진행/완료 후 기록 |

### Step 1: 단일변수 Ablation (공식 해상도 960, Exp19-A~D)

> [!CAUTION]
> **[설계 수정 — 2026-03-31]** 초기 Step 1 설계(Exp19-A: auto_augment, Exp19-B: erasing)는 폐기한다.
> Ultralytics 8.4.23 소스코드(`augment.py`) 직접 검증 결과, `auto_augment`와 `erasing`은 **detect 파이프라인(`v8_transforms`)에서 호출되지 않는 no-op 파라미터**임이 확인됨. (Codex 분석 + 소스코드 교차 확인)
> 이 두 파라미터를 튜닝하는 것은 실험 슬롯과 GPU 시간만 낭비하는 행위. 즉시 폐기하고 실제 작동하는 파라미터로 대체.

기준은 **Exp15(960) 설정**이며, **`v8_transforms`에서 실제로 호출이 확인된** 파라미터 4개만 단독 변경한다.

| 실험 ID | 변경 파라미터 | 기준(Exp15) → 실험값 | 가설 | 작동 위치 |
|:---:|:---|:---:|:---|:---|
| **Exp 19-A** | `hsv_h` | `0.015` → `0.0` (OFF) | 색조 변환이 알약 색상 기반 구분을 방해할 가능성 | `RandomHSV` |
| **Exp 19-B** | `mosaic` | `1.0` → `0.5` | 4-패치 합성 시 소형 알약 과도한 절단 완화 | `Mosaic` |
| **Exp 19-C** | `translate` | `0.1` → `0.0` (OFF) | 위치 이동이 정형화된 센터 촬영 환경에서 불필요할 수 있음 | `RandomPerspective` |
| **Exp 19-D** | `scale` | `0.5` → `0.3` (완화) | 크기 변환이 알약 비율 왜곡을 유발할 수 있음 | `RandomPerspective` |

> **Step 1 판정 기준(960 기준)**: Exp15 Kaggle(0.96455) 대비 **상승폭 우선**으로 top2 선별.

### Step 2: Top2 조합 검증 (경량 설계, Exp20)

Step 1에서 top2를 `A`, `B`로 고른 뒤, **Step1의 `A only`, `B only` 결과를 재사용**하고 조합 실험만 추가한다.

| 실험 ID | 설정 | 목적 |
|:---:|:---|:---|
| **Exp 20-AB** | A + B | 시너지/충돌 확인(필수) |
| **Exp 20-Base (옵션)** | Baseline (변경 없음) | 필요 시 동일 시점 재현성 체크 |

> **해석 규칙(경량)**
> - `AB > max(A, B)`: 시너지  
> - `AB <= max(A, B)`: 조합 이득 없음, 단일 설정 우선  
> - 즉, 시간 절약을 위해 `A/B` 재실행은 생략하고 `AB`만으로 조합 채택 여부를 판단

### Step 2.5: YOLO 내장 `copy_paste` 게이트 (보류)

현재 데이터셋 라벨은 `class x y w h` 5컬럼 bbox만 존재(세그멘트 마스크 부재)하여,  
Ultralytics `CopyPaste.__call__()`에서 `len(labels["instances"].segments) == 0` 조건으로 즉시 `return`된다 (`augment.py` L.1713).

| 실험 ID | 상태 | 재개 조건 |
|:---:|:---|:---|
| **Exp 20-CP03/CP05** | **보류** | 세그멘테이션 폴리곤 라벨 확보 후 재개 |

### Step 3: 1024 확장 검증 (비차단, Exp21)

Step 2까지 통과한 최종 후보를 `1024`에서 확장 검증한다.
이 단계는 **공식 960 제출 트랙을 막지 않는 비차단 단계**로 운영한다.

| 실험 ID | 내용 | 비고 |
|:---:|:---|:---|
| **Exp 21-Base** | Exp17(1024) 기준선 재확인 | 확장 트랙 기준선 |
| **Exp 21-Candidate** | Step 2 최적 후보를 1024에 적용 | 1024 확장 채택 판단 |

### Step 4: Multi-Seed 최종화 (Exp22-S)

| 실험 ID | 내용 | 비고 |
|:---:|:---|:---|
| **Exp 22-S** | 최종 설정으로 Seed 42/123/777 학습 → WBF 앙상블 | Exp 10(F) 방식 재현, 목표 ≥0.985 |

---

## 4. 📊 지표 관리 및 성공 기준

1. **Kaggle Public Score**: 최우선 지표. 목표: 단일 모델 ≥ 0.970
2. **Local mAP@50-95**: Exp 12(0.9933) 이하로 떨어지지 않는지 확인. 단, 데이터 누수(Leakage)로 Local과 Kaggle 간 괴리가 크므로 Local만으로 판단 금지.
3. **해석 매트릭스**:

| Local mAP | Kaggle Score | 해석 |
|:---:|:---:|:---|
| ↑ | ↑ | ✅ 정답. 일반화 성능 향상. |
| ↑ | ↓ | ⚠️ 과적합 심화. 증강이 부족하거나 잘못됨. |
| ↓ | ↑ | 🔵 증강이 과적합을 깨뜨림. 좋은 신호. |
| ↓ | ↓ | 🚫 해당 변경은 완전히 해로움. 즉시 롤백. |

---

## 5. 🔵 YOLO 내장 `copy_paste` vs 커스텀 Copy-Paste 분석

> [!WARNING]
> 이 두 가지는 **완전히 다른 메커니즘**이다. 혼동하면 실험 설계가 엉망이 된다.

### 비교표

| 항목 | 커스텀 Copy-Paste (Exp 5) | YOLO 내장 `copy_paste` |
|:---|:---|:---|
| **작동 시점** | **오프라인 (전처리)**. 학습 전에 물리적 이미지 파일 317장을 생성하여 데이터셋에 추가. | **온라인 (학습 중)**. 매 배치마다 실시간으로 한 이미지의 객체를 다른 이미지 위에 합성. |
| **구현 위치** | `preprocessing.py` → `apply_copy_paste_augmentation()` | Ultralytics 내부 (`augment.py`) |
| **타겟** | **희귀 클래스만** 선별 증강 (target_count=20). Pill Bank에서 크롭 후 연회색 캔버스에 배치. | **전체 클래스 무차별 적용**. 확률(`copy_paste` 값)에 따라 랜덤 객체를 합성. |
| **세그멘테이션 마스크** | 불필요 (BBox 직접 합성) | **필수** (마스크 기반으로 객체를 정밀하게 잘라 붙임). 마스크 없으면 사실상 작동 안 함. |
| **검증 결과** | Kaggle **0.701 → 0.968** (+0.267). 폭발적 효과 검증 완료. | 미검증. |

### ❗ YOLO 내장 `copy_paste` 활성화 시 주의사항

> YOLO의 내장 copy_paste는 **세그멘테이션 마스크(segmentation annotation)**가 있어야 제대로 작동한다. 마스크가 없으면 BBox 영역 전체를 잘라 붙이는데, 이 경우 배경까지 함께 복사되어 **오히려 성능을 해칠 수 있다.**

현재 프로젝트의 YOLO 라벨 파일은 train/val 모두 **전량 5컬럼 bbox 포맷**으로 확인되어, 내장 `copy_paste`는 현시점에서 사실상 no-op다.

### 실험 고려사항

| 시나리오 | 권장 |
|:---|:---|
| segmentation 마스크가 **유효**함 | `copy_paste=0.3~0.5` 재검토 가능. 단, 커스텀 CP와 동시 적용 시 과증강 리스크 점검. |
| **현재 상태: bbox-only(5컬럼)** | `copy_paste=0.0` 유지, 실험 보류. |

---

## 6. 🎯 Rotation 재실험 판단 (Exp 6 회고)

| 항목 | Exp 6에서 한 것 | YOLO 디폴트 |
|:---|:---|:---|
| `degrees` | **180.0** (전방위 회전) | **0.0** (회전 없음) |
| 효과 | Kaggle **0.968 → 0.854** (-0.114) | - |

**Exp 6은 YOLO "기본 증강"이 아니다.** `degrees=180.0`은 사실상 flip과 동일한 파괴 효과다.
다만, "소각도 회전(`degrees=5~10`)"은 카메라 앵글 오차를 모사할 수 있어 이론적으로 유효하다.

> **결론**: 현재 도메인(정방향 촬영, 연회색 배경)에서 회전의 이득은 **불확실**하다. Step 1~2 결과를 본 뒤, 여유가 있으면 `degrees=5.0`을 Step 3에서 시도하는 정도로 후순위 배치.

---

## 7. ⏰ 실행 우선순위 타임라인

```text
[마감] Phase 0: Exp18(1280) 해상도 벤치마크 완료/기록
  ↓
[즉시] Step 1: 960 단일변수 4개 Ablation (Exp19-A~D)
  ↓ top2 선별
[다음] Step 2: Top2 조합 검증 (Exp20-AB, 필요시 Base 재검증)
  ↓ 960 공식 후보 확정
[병행/여유] Step 3: 1024 확장 검증 (Exp21-Base/Candidate)
  ↓ 확장 트랙 채택 여부 판단
[최종] Step 4: 3-Seed 재검증 + WBF 앙상블 (Exp22-S)
  ↓ (여유 시)
[보류] Step 2.5: YOLO copy_paste 게이트 (세그멘트 라벨 확보 후)
[후순위] degrees=5.0 소각도 회전
[팀원 담당] CLAHE / box_weight / cls_weight 조정
```

> [!TIP]
> **핵심을 잊지 마라**: 지금 네 모델의 가장 큰 점수 점프는 증강 튜닝이 아니라 **앙상블(Exp 10: +0.009)**과 **커스텀 Copy-Paste(Exp 5: +0.267)**에서 왔다. 온라인 증강 튜닝은 "독을 빼서 0.002~0.006을 챙기는" 마무리 작업이다. 과한 기대는 금물.
