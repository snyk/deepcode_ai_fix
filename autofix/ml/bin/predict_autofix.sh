#!/bin/bash
if [[ -z "${MODEL_NAME}" ]]; then
    echo "please provide a model name in the MODEL_NAME env variable"
    exit 1
fi
LAST_MODEL_NAME_COMPONENT=$(echo "$MODEL_NAME" | awk -F/ '{print $NF}')

if [[ -z "${MAX_NUM_TOKENS}" ]]; then
    echo "please provide the tokenizer max length in the MAX_NUM_TOKENS env variable"
    exit 1
fi

if [[ -z "${BATCH_SIZE}" ]]; then
    echo "please provide the batch size in the BATCH_SIZE env variable"
    exit 1
fi

NUM_BEAMS=5

export TRANSFORMERS_NO_ADVISORY_WARNINGS="True"

dc_ai_fix_venv/bin/python3 autofix/ml/bin/predict_autofix.py \
    --output_dir="/tmp/predict_autofix" \
    --per_device_eval_batch_size="${BATCH_SIZE}" \
    --data_id="data/paper_test.parquet" \
    --model_id="t5-small" \
    --model_name="${LAST_MODEL_NAME_COMPONENT}" \
    --model_id="${MODEL_NAME}" \
    --beam_size="${NUM_BEAMS}" \
    --num_return_seqs="${NUM_BEAMS}" \
    --tokenizer_max_length="${MAX_NUM_TOKENS}" \
    --max_new_tokens="${MAX_NUM_TOKENS}" \
    --seed=42 \
    --dataloader_num_workers=10 \
    --bf16_full_eval="True" \
    --torch_compile_mode="max-autotune" \
    --preprocessing_batch_size=1000 \
