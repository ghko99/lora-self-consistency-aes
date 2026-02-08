# LLaMA-AES: Advanced Automated Essay Scoring with Dynamic Loss Weighting

본 프로젝트는 **LLaMA-3.1** 모델을 기반으로 자동 에세이 채점(AES) 성능을 극대화하기 위해 개발되었습니다. 단순한 Cross-Entropy Loss를 넘어, 점수 예측의 정교함을 높이는 NTL(Number Token Loss)과 의미적 유사도를 반영하는 EMO(Embedding-based Metric-Oriented Loss)를 결합하여 학습합니다.

특히, 학습 과정에서 손실 함수 간의 스케일 차이를 자동으로 보정하는 **Dynamic Loss Weighting** 기법을 적용하여 최적의 성능을 달성했습니다.

## 📌 주요 특징 (Key Features)

1. **Dynamic Loss Weighting (성능 핵심)**
* 서로 다른 손실 함수(CE, NTL, EMO) 간의 크기 차이로 인한 학습 불균형을 막기 위해, 매 Step마다 손실 비율에 따라 가중치를 동적으로 조정합니다.
* 고정된 가중치(Static Weight)를 사용하는 것보다 수렴이 빠르고 최종 성능(QWK)이 더 우수하게 나타났습니다.


2. **복합 손실 함수 (Composite Loss)**
* **CE (Cross-Entropy):** 기본적인 언어 모델링 학습.
* **NTL (Number Token Loss):** 숫자 토큰의 기댓값(Expectation)을 계산하여 정답 점수와의 오차를 최소화합니다 (MSE / Wasserstein).
* **EMO (Embedding-based Metric-Oriented Loss):** 모델이 예측한 Top-K 토큰들의 임베딩 가중 평균과 정답 임베딩 간의 코사인 유사도를 학습에 반영합니다.


3. **멀티 태스크 학습 (Multi-Task Learning, MTL)**
* 에세이 채점(Score)과 피드백 생성(Feedback)을 동시에 학습합니다.
* MTL 모드 시 점수 부분은 NTL로, 피드백 부분은 EMO로 최적화됩니다.


4. **Self-Consistency 분석**
* 하나의 프롬프트에 대해 다수의 응답(개)을 생성하고, **Majority Vote** 및 **Average Vote**를 통해 예측 신뢰도를 분석합니다.



---

## 📂 프로젝트 구조 (Project Structure)

```
/
├── main_pipeline.py             # [학습 -> 추론 -> 평가] 전체 파이프라인 실행
├── self_consistency.py          # Self-Consistency 샘플링 및 분석 도구
├── requirements.txt             # 의존성 패키지 목록
├── aes_dataset_mtl/             # 학습 데이터셋 (Train/Valid/Test)
└── modules/
    ├── aes_dataloader.py        # 데이터 전처리 및 Collator (MTL 지원)
    ├── custom_trainer.py        # Dynamic Weighting이 구현된 Trainer
    ├── number_token_loss.py     # MSE 기반 NTL 구현
    ├── wasserstein_number_token_loss.py # Wasserstein 기반 NTL 구현
    ├── inference_module.py      # 모델 추론 및 CSV 저장
    └── evaluate_module.py       # QWK 점수 계산 및 평가

```

---

## 🛠️ 설치 방법 (Installation)

**Python 3.10+** 환경에서 실행하는 것을 권장합니다.

```bash
# 1. 가상환경 생성 (예시)
conda create -n aes_env python=3.10 -y
conda activate aes_env

# 2. 패키지 설치
pip install -r requirements.txt

```

---

## 🚀 사용 방법 1: 모델 학습 (Training Pipeline)

`main_pipeline.py`는 학습(Train), 추론(Inference), 평가(Evaluation)를 순차적으로 수행합니다.

### 🔥 추천: Dynamic Weighting 모드 실행 (Best Performance)

가중치 인자에 **음수 값(예: -1.0)**을 주면 Dynamic Weighting이 활성화됩니다.

```bash
python main_pipeline.py \
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --mtl \
    --ntl_weights -1.0 \
    --emo_weights -1.0 \
    --loss_type mse \
    --device_id 0

```

* `--ntl_weights -1.0`: NTL 손실에 대해 동적 가중치 적용
* `--emo_weights -1.0`: EMO 손실에 대해 동적 가중치 적용 (NTL+EMO 통합 동적 조절)

### 일반 실행 (고정 가중치)

```bash
python main_pipeline.py \
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --mtl \
    --ntl_weights 2.0 \
    --emo_weights 0.1 \
    --device_id 0

```

### 주요 인자 설명

| 인자 | 설명 | 기본값 |
| --- | --- | --- |
| `--base_model_name` | (필수) 베이스 모델 경로 또는 HuggingFace ID | - |
| `--mtl` | Multi-Task Learning 데이터셋 사용 여부 | `False` |
| `--ntl_weights` | NTL 가중치. **음수 입력 시 Dynamic Weighting 활성화** | `2.0` |
| `--emo_weights` | EMO 가중치. **음수 입력 시 Dynamic Weighting 활성화** | `0.1` |
| `--loss_type` | NTL 손실 함수 종류 (`mse` 또는 `was`) | `mse` |
| `--ratio` | 학습 데이터 사용 비율 (0.1 = 10%) | `1.0` |
| `--resume_checkpoint` | 학습 재개할 체크포인트 경로 | `None` |

---

## 📊 사용 방법 2: Self-Consistency 분석

`self_consistency.py`는 학습된 모델을 사용하여 샘플링 및 앙상블 효과를 분석합니다.

### 1. Run 모드 (샘플링 + 분석)

```bash
python self_consistency.py run \
    --adapter_dir "./runs/your_best_adapter" \
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --test_path "./aes_dataset_mtl/test.jsonl" \
    --max_m 50 \
    --temperature 0.7 \
    --device_id 0

```

### 2. Analyze 모드 (분석 전용)

이미 생성된 샘플 JSON 파일이 있을 때 그래프만 다시 그립니다.

```bash
python self_consistency.py analyze \
    --bank_path "./consistency_results/.../samples_m50_xxxx.json" \
    --test_path "./aes_dataset_mtl/test.jsonl"

```

---

## 📝 데이터셋 포맷 (JSONL)

**MTL 포맷 예시:**

```json
{
  "instruction": "Evaluate the following essay...",
  "output": "4 3 4 3 4 3 4 3\n\nEssay Feedback: The essay shows..."
}

```

* `custom_trainer.py`는 `output`의 **숫자 점수 부분**과 **텍스트 피드백 부분**을 자동으로 구분하여 각각 NTL과 EMO Loss를 적용합니다.
