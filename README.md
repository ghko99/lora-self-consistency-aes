# AES 모델 파인튜닝 및 Self-Consistency 분석 프로젝트

## 1. 개요 (Overview)

본 프로젝트는 특정 Task(Automated Essay Scoring, AES)에 대해 LLaMA 기반 언어 모델을 파인튜닝하고, Self-Consistency 기법을 통해 모델의 일관성을 평가 및 분석하는 것을 목표로 합니다.

주요 기능은 다음과 같습니다.

- **모델 파인튜닝**: LoRA(Low-Rank Adaptation)를 사용하여 주어진 데이터셋으로 언어 모델을 학습시킵니다.
- **Self-Consistency 분석**: 학습된 모델을 사용하여 하나의 프롬프트에 대해 여러 개의 답변을 생성하고, 이를 집계(majority/average voting)하여 QWK(Quadratic Weighted Kappa) 점수 변화를 측정하고 시각화합니다.

---

## 2. 동작 원리 (How it Works)

### 2.1. 학습 과정 (`main_pipeline.py`)

1.  **모델 로딩**: `meta-llama/Llama-3.1-8B-Instruct`와 같은 사전 학습된 언어 모델(Base Model)을 불러옵니다.
2.  **LoRA 적용**: 모델의 가중치를 직접 수정하는 대신, LoRA(Low-Rank Adaptation) 어댑터를 추가하여 파라미터 효율적인 파인튜닝(PEFT)을 수행합니다. 이를 통해 적은 양의 파라미터만으로 모델을 특정 태스크에 맞게 조정할 수 있습니다.
3.  **커스텀 학습**: `CustomTrainer`는 복합적인 손실 함수(Composite Loss)를 사용하여 모델을 학습시킵니다.
    - **CE (Cross-Entropy) Loss**: 일반적인 언어 모델의 기본 손실 함수입니다.
    - **NTL (Number Token Loss)**: 모델이 예측한 숫자 토큰들의 기댓값과 실제 정답 숫자 사이의 오류(MSE 또는 Wasserstein)를 계산하여, 점수 예측의 정확성을 높입니다.
    - **EMO (Embedding-based Metric-Oriented) Loss**: 정답 임베딩과 모델이 예측한 상위 K개 토큰의 가중 평균 임베딩 사이의 코사인 유사도를 손실에 반영하여, 의미적으로 더 가까운 예측을 하도록 유도합니다.
4.  **모델 저장**: 학습이 완료되면, 원본 모델의 가중치는 그대로 둔 채 학습된 LoRA 어댑터의 가중치만 `./runs/` 디렉터리에 저장됩니다.

### 2.2. Self-Consistency 분석 과정 (`self_consistency.py`)

1.  **모델 로딩**: 사전 학습된 베이스 모델 위에, 지정된 LoRA 어댑터를 결합하여 파인튜닝된 모델을 불러옵니다.
2.  **다중 샘플 생성**: 테스트 데이터셋의 각 프롬프트에 대해, `do_sample=True` 옵션으로 **m**개의 서로 다른 답변을 생성합니다. (e.g., m=50)
3.  **답변 집계**: 생성된 m개의 답변에서 점수(숫자)를 추출하고, 두 가지 방식으로 최종 점수를 집계합니다.
    - **Average Vote**: 유효한 점수들의 평균을 계산한 후 반올림합니다.
    - **Majority Vote**: 가장 많이 등장한 점수를 최종 예측 점수로 선택합니다.
4.  **QWK 점수 계산 및 시각화**: 샘플 개수(m)를 1부터 최댓값까지 늘려가면서, 각 m값에 대한 Average/Majority Vote 예측 점수와 실제 정답 간의 QWK(Quadratic Weighted Kappa) 점수를 계산합니다.
5.  **결과 저장**: 계산된 점수 커브를 그래프(png)로 시각화하고, 분석에 사용된 샘플 데이터와 최종 점수를 JSON 파일로 `./consistency_results/` 디렉터리에 저장합니다.

---

## 3. 프로젝트 구조 (Project Structure)

```
/
├─── main_pipeline.py             # 모델 학습, 추론, 평가를 실행하는 메인 스크립트
├─── self_consistency.py          # Self-Consistency 샘플링 및 분석/시각화를 실행하는 스크립트
├─── requirements.txt             # 프로젝트 의존성 패키지 목록
├─── aes_dataset_mtl/             # 학습 및 테스트에 사용되는 데이터셋
└─── modules/                     # 주요 기능을 구현한 모듈
    ...
```

---

## 4. 설치 방법 (Setup)

### 4.1. 가상환경 생성 (Anaconda)

본 프로젝트는 **Python 3.10** 버전에서 테스트되었습니다. Anaconda를 사용하여 다음과 같이 가상환경을 생성하고 활성화합니다.

```bash
# 1. 가상환경 생성
conda create --name aes_env python=3.10 -y

# 2. 가상환경 활성화
conda activate aes_env
```

### 4.2. 의존성 패키지 설치

`requirements.txt` 파일을 사용하여 필요한 모든 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

---

## 5. 사용 방법 (Usage)

실행 전, `conda activate aes_env` 명령어로 가상환경이 활성화되었는지 확인하세요.

### 5.1. 모델 학습 (`main_pipeline.py`)

`main_pipeline.py`는 모델을 파인튜닝하고, 그 결과를 바탕으로 추론 및 평가를 수행합니다.

**실행 예시:**

```bash
python main_pipeline.py 
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" 
    --device_id 0 
    --mtl 
    --no_wandb
```

**주요 실행 인자:**

| 인자                  | 설명                                                                                                         | 기본값 |
| --------------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| `--base_model_name`   | **(필수)** 파인튜닝의 기반이 될 모델의 경로 또는 Hugging Face Hub 이름. (예: `meta-llama/Llama-3.1-8B-Instruct`) | -      |
| `--device_id`         | 학습에 사용할 GPU 장치 ID.                                                                                   | -      |
| `--mtl`               | Multi-Task Learning용 데이터셋(`aes_dataset_mtl`)을 사용하려면 이 플래그를 추가합니다.                         | `False`  |
| `--no_wandb`          | Weights & Biases 로깅을 비활성화합니다.                                                                      | `False`  |
| `--dry_run`           | 실제 학습 없이, 모델/데이터 로딩 등 설정이 올바른지만 확인합니다.                                            | `False`  |
| `--ratio`             | 전체 데이터셋 중 사용할 비율을 지정합니다. (0.1 = 10%)                                                       | `1.0`    |
| `--ntl_weights`       | NTL(Number Token Loss)의 가중치를 설정합니다.                                                                | `2.0`    |
| `--emo_weights`       | EMO(Embedding-based Metric-Oriented) Loss의 가중치를 설정합니다.                                             | `0.1`    |
| `--loss_type`         | NTL의 손실 함수 유형을 선택합니다. (`mse` 또는 `was`)                                                        | `mse`    |
| `--resume_checkpoint` | 지정된 경로의 체크포인트에서 학습을 재개합니다.                                                              | `None`   |


### 5.2. Self-Consistency 분석 (`self_consistency.py`)

`self_consistency.py`는 학습된 모델의 일관성을 분석합니다. `run` 모드를 사용하여 샘플링부터 분석, 시각화까지 한 번에 실행합니다.

**실행 예시:**

```bash
python self_consistency.py run 
    --adapter_dir "./runs/ntl_emo" 
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" 
    --test_path "./aes_dataset_mtl/test.jsonl" 
    --device_id 0
```

**주요 실행 인자 (`run` 모드):**

| 인자                | 설명                                                                                         | 기본값                  |
| ------------------- | -------------------------------------------------------------------------------------------- | ----------------------- |
| `--adapter_dir`     | **(필수)** 분석할 학습된 LoRA 어댑터 모델이 저장된 경로.                                     | -                       |
| `--base_model_name` | **(필수)** 기반이 되는 모델의 경로 또는 Hugging Face Hub 이름.                               | -                       |
| `--test_path`       | **(필수)** 평가에 사용할 테스트 데이터(`.jsonl`)의 경로.                                     | -                       |
| `--device_id`       | 샘플링에 사용할 GPU 장치 ID.                                                                 | `0`                     |
| `--dry_run`         | 실제 샘플링 없이, 설정이 올바른지만 확인합니다.                                              | `False`                   |
| `--max_m`           | 프롬프트 당 생성할 최대 샘플 수.                                                             | `50`                    |
| `--chunk_m`         | VRAM 관리를 위한 생성 chunk 크기. (VRAM 부족 시 줄여서 사용)                                 | `10`                    |
| `--temperature`     | 샘플링 시 사용할 temperature 값. (높을수록 다양성 증가)                                      | `0.7`                   |
| `--output_root`     | 결과가 저장될 최상위 기본 디렉터리.                                                          | `./consistency_results` |
