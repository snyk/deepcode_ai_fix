#!/bin/bash
if [[ -z "${MODEL_NAME}" ]]; then
    echo "please provide a model name in the MODEL_NAME env variable"
    exit 1
fi
LAST_MODEL_NAME_COMPONENT=$(echo "$MODEL_NAME" | awk -F/ '{print $NF}')

if [[ -z "${NUM_EPOCHS}" ]]; then
    echo "please provide the number of epochs in the NUM_EPOCHS env variable"
    exit 1
fi

if [[ -z "${INPUT_MAX_NUM_TOKENS}" ]]; then
    echo "please provide the tokenizer max length in the INPUT_MAX_NUM_TOKENS env variable"
    exit 1
fi

OUTPUT_DIR="/tmp/train_autofix"

# These disable the annoying warning messages from HF.
export TRANSFORMERS_NO_ADVISORY_WARNINGS="True"
export TOKENIZERS_PARALLELISM="False"

dc_ai_fix_env/bin/python3 autofix/ml/bin/train_autofix.py \
    --output_dir="${OUTPUT_DIR}" \
    --model_name="${MODEL_NAME}" \
    --num_train_epochs="${NUM_EPOCHS}" \
    --learning_rate="1e-5" \
    --warmup_ratio=0.1 \
    --save_strategy="epoch" \
    --save_total_limit=1 \
    --evaluation_strategy="epoch" \
    --load_best_model_at_end="True" \
    --greater_is_better="False" \
    --metric_for_best_model="eval_loss" \
    --ddp_find_unused_parameters="False" \
    --remove_unused_columns="False" \
    --logging_steps=1 \
    --seed=42 \
    --data_ids="fixdb-license-all-js" \
    --data_tags="2023-09-11T113827.392" \
    --tokenizer_max_length="${INPUT_MAX_NUM_TOKENS}" \
    --max_new_tokens="${INPUT_MAX_NUM_TOKENS}" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps=16 \
    --model_artifact="llm-trials/${LAST_MODEL_NAME_COMPONENT}" \
    --bf16="True" \
    --preprocessing_batch_size=1000 \
    --torch_compile_mode="max-autotune" \
    --deepspeed="autofix/ml/bin/deepspeed_config.json" \
    --dataloader_num_workers=10
