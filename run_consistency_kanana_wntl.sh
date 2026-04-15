#!/bin/bash

# Configuration
PROJECT_ROOT="/shared/home/aif/kkh_1"
SCRIPT_DIR="${PROJECT_ROOT}/lora-self-consistency-aes"
SCRIPT_PATH="${SCRIPT_DIR}/self_consistency.py"
ADAPTER_DIR="${PROJECT_ROOT}/aes_llm_training/runs/kanana_wntl_20260407_002343"
BASE_MODEL="/shared/home/aif/hf_models/kanana"
DATASET_DIR="${SCRIPT_DIR}/aes_dataset_mtl"
OUTPUT_ROOT="${SCRIPT_DIR}/consistency_results"

# Self-consistency parameters
MAX_M=50
CHUNK_M=10
TOP_K=9
TEMPERATURE=0.7
DEVICE_ID=0

for SPLIT in 1 2 3; do
    TEST_PATH="${DATASET_DIR}/test_14_${SPLIT}.jsonl"
    TAG="kanana_wntl_test_14_${SPLIT}_m${MAX_M}"

    echo "=========================================="
    echo "Running self-consistency for test_14_${SPLIT}.jsonl"
    echo "Adapter: ${ADAPTER_DIR}"
    echo "Test:    ${TEST_PATH}"
    echo "Tag:     ${TAG}"
    echo "=========================================="

    python3 "${SCRIPT_PATH}" run \
        --adapter_dir "${ADAPTER_DIR}" \
        --base_model_name "${BASE_MODEL}" \
        --test_path "${TEST_PATH}" \
        --max_m ${MAX_M} \
        --chunk_m ${CHUNK_M} \
        --top_k ${TOP_K} \
        --temperature ${TEMPERATURE} \
        --device_id ${DEVICE_ID} \
        --output_root "${OUTPUT_ROOT}" \
        --tag "${TAG}" \
        --fallback_score 5 \
        --xtick_step 10
done

echo "All splits complete. Results are in ${OUTPUT_ROOT}"
