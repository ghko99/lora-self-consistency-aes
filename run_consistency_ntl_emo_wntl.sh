#!/bin/bash

# Configuration
PROJECT_ROOT="/home/khko/workspace/Automated_Essay_Scoring"
SCRIPT_PATH="${PROJECT_ROOT}/lora-self-consistency-aes/self_consistency.py"
ADAPTER_DIR="${PROJECT_ROOT}/paper_models/ntl_emo_wntl_20260306_204610_mse"
BASE_MODEL="/home/khko/models/llama"
TEST_PATH="${PROJECT_ROOT}/paper_models/test.jsonl"
OUTPUT_ROOT="${PROJECT_ROOT}/lora-self-consistency-aes/consistency_results"

# Self-consistency parameters
MAX_M=50
CHUNK_M=10
TOP_K=9
TEMPERATURE=0.7
DEVICE_ID=0

echo "Starting self-consistency analysis for model: ntl_emo_wntl_20260306_204610_mse"
echo "Parameters: m=${MAX_M}, temp=${TEMPERATURE}, top_k=${TOP_K}"

# Run the script
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
    --tag "ntl_emo_wntl_m50" \
    --fallback_score 5 \
    --xtick_step 10

echo "Analysis complete. Results are in ${OUTPUT_ROOT}"
