#!/bin/bash

if [[ -z "${MODEL_NAME}" ]]; then
    echo "please provide a model name in the MODEL_NAME env variable"
    exit 1
fi

if [[ -z "${DATA_ID}" ]]; then
    echo "please provide the data id in the DATA_ID env variable"
    exit 1
fi

if [[ -z "${NUM_PREDICTIONS}" ]]; then
    echo "please provide the number of predictions to generate in the NUM_PREDICTIONS env variable"
    exit 1
fi

if [[ -z "${TEMPERATURE}" ]]; then
    echo "please provide the temperature in the TEMPERATURE env variable"
    exit 1
fi

if [[ -z "${NUM_SHOTS}" ]]; then
    echo "please provide the num_shots in the NUM_SHOTS env variable"
    exit 1
fi

if [[ -z "${MAX_NEW_TOKENS}" ]]; then
    echo "please provide the max_new_tokens in the MAX_NEW_TOKENS env variable"
    exit 1
fi

dc_ai_fix_env/bin/activate/python3 autofix/ml/bin/predict_llm.py \
    --model_name="${MODEL_NAME}" \
    --data_id="${DATA_ID}" \
    --num_predictions="${NUM_PREDICTIONS}" \
    --temperature="${TEMPERATURE}" \
    --num_shots="${NUM_SHOTS}" \
    --max_new_tokens="${MAX_NEW_TOKENS}"
