import os
import tempfile

from typing import cast
import pandas as pd
import peft
import torch
import transformers
from datasets.arrow_dataset import Dataset
from pandera.typing import DataFrame
from peft.utils.other import prepare_model_for_kbit_training
from tokenizers import AddedToken
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers import T5TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

import autofix.ml.lib.args as arguments
import autofix.ml.lib.data_processor as data_processor
import autofix.ml.utils.loading as ld
from autofix.ml.lib.data_schemas import LabelledDataSchema
from autofix.ml.lib.trainer import AutofixTrainer
from ml_utils.distributed_training import MultiProcessLogger
from ml_utils.distributed_training import is_main_process
from ml_utils.model_type import ModelType

logger = MultiProcessLogger()
transformers.logging.set_verbosity_info()


def _patch_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
    if isinstance(tokenizer, T5TokenizerFast):
        logger.info(
            "Adding code-related tokens to the T5 tokenizer",
            model_name_or_path=model.config.name_or_path,
        )
        tokenizer.add_tokens(["{", "}", "<", ">", "\\", "^", "`", "~"])
        tokenizer.add_tokens(AddedToken("\n", normalized=False))  # type: ignore
        tokenizer.add_tokens(AddedToken("\t", normalized=False))  # type: ignore
        tokenizer.add_tokens(AddedToken("\r\n", normalized=False))  # type: ignore
        model.resize_token_embeddings(len(tokenizer))

    if model.config.model_type in [
        "gpt2",
        "gpt_bigcode",
        "mosaic_gpt",
        "llama",
        "mpt",
        "mixtral",
        "stablelm_epoch",
        "starcoder2",
    ]:
        # These models (e.g. GPT2) don't use padding tokens. To make it work with the
        # padding data collators, we set the padding token to unknown token, if defined, or else to the end of sequence token.
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model.pad_token_id = tokenizer.pad_token_id  # type: ignore
            model.config.pad_token_id = tokenizer.pad_token_id

            logger.info(
                "Setting the pad_token_id for a decoder model",
                new_pad_token_id=tokenizer.pad_token_id,
                new_pad_token=tokenizer.pad_token,
                model_pad_token_id=model.pad_token_id,
                model_config_pad_token_id=model.config.pad_token_id,
                model_name_or_path=model.config.name_or_path,
            )
            assert tokenizer.pad_token_id is not None
            assert (
                model.pad_token_id == tokenizer.pad_token_id
            ), "The model's pad token id and the tokenizer's pad token id are not the same"

    # Handle the chat template.
    if tokenizer.chat_template is None:
        old_tokenizer_len = len(tokenizer)
        tokenizer.add_tokens(data_processor.make_special_tokens(), special_tokens=True)  # type: ignore
        logger.info(
            "Adding new special tokens",
            num_new_special_tokens=len(tokenizer) - old_tokenizer_len,
            new_num_tokens=len(tokenizer),
        )
        model.resize_token_embeddings(len(tokenizer))


def train_autofix() -> None:
    # Step 0 - setup.
    args: arguments.AutofixArgs = arguments.parse_autofix_args_from_cmd()
    transformers.trainer_utils.set_seed(args.training_args.seed)
    logger.info("Set transformers seed.", seed=args.training_args.seed)
    if not args.model_args.model_id:
        raise RuntimeError("model_id can not be empty or None.")

    best_model_path = tempfile.TemporaryDirectory()

    # Step 1 - fetch the data.
    with args.training_args.main_process_first():
        train_df, val_df = pd.read_parquet("train.parquet"), pd.read_parquet("validation.parquet")

    train_ds = Dataset.from_pandas(train_df)
    logger.info("Training size", num=len(train_ds))

    val_ds = Dataset.from_pandas(val_df)
    logger.info("Validation size", num=len(val_ds))

    # Step 2 - fetch the model and the tokenizer
    model_loading_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "use_flash_attention_2": args.model_args.use_flash_attention_2,
    }
    if args.model_args.use_bits_and_bytes:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_loading_kwargs.update({"quantization_config": bnb_config})
    if args.training_args.deepspeed is None:
        model_loading_kwargs.update({"device_map": "auto"})

    model = ld.load_suitable_model(
        model_identifier=args.model_args.model_name,
        **model_loading_kwargs,
    )

    if args.model_args.use_peft:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.training_args.gradient_checkpointing
        )
        lora_config = peft.tuners.lora.LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "lm_head",
            ],
            lora_dropout=0.05,
            bias="all",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        model = peft.mapping.get_peft_model(model, lora_config)
        logger.info(
            "Created peft the model",
            config=lora_config
        )
        model.print_trainable_parameters()

    tokenizer = ld.load_suitable_tokenizer(args.model_args.model_name)
    _patch_token_ids(model, tokenizer)  # type: ignore

    # Step 3 - preprocess the datasets.
    model_type = ModelType.infer_from_model(model)  # type: ignore

    def process_batch(batch: DataFrame[LabelledDataSchema]) -> BatchEncoding:
        prompts = data_processor.make_prompts(
            batch, model_type, tokenizer, include_post=True
        )
        return data_processor.tokenize_prompts(
            prompts=prompts,
            labels=cast(list[str], batch[LabelledDataSchema.post_reduced]),
            tokenizer=tokenizer,
            truncation=True,
            max_length=args.model_args.tokenizer_max_length,
            model_type=model_type,
        )

    def process_dataset(ds: Dataset, ds_name: str) -> Dataset:
        return ds.map(
            process_batch,
            batched=True,
            batch_size=args.dataset_args.preprocessing_batch_size,
            keep_in_memory=args.dataset_args.keep_in_memory,
            desc=f"Processing the {ds_name} dataset",
            remove_columns=ds.column_names,
        )

    train_ds = process_dataset(train_ds, "training")
    val_ds = process_dataset(val_ds, "validation")

    # Step 4 - train the model.
    logger.info("Starting with autofix training...")
    trainer = AutofixTrainer(
        model=model,  # type: ignore
        tokenizer=tokenizer,
        args=args,
        train_ds=train_ds,
        val_ds=val_ds,
    )
    trainer.train()

    # Step 5 - save the trained model and tokenizer
    logger.info("saving the best model", best_model_path=best_model_path)
    trainer.save_model(str(best_model_path))

    if trainer.is_deepspeed_enabled:
        # If we're training with deepspeed and ZERO-3, then we additionally need to convert
        # the bf16/fp16 weights from deepspeed's checkpoints into a fp32 torch state_dict,
        # that we can later use for inference.

        # `trainer.deepspeed` is actually the model object (wrapped into a deepspeed wrapper)
        assert trainer.deepspeed is not None

        # This saves the current deepspeed checkpoint into `best_checkpoint_dir`.
        # If `--load_best_model_at_end` is specified, this checkpoint will correspond
        # to the best checkpoint. Otherwise, it will correspond to the current state of the model.
        checkpoint_tag = "export-checkpoint"
        logger.info(
            "saving the best deepspeed checkpoint",
            path=args.training_args.output_dir,
            tag=checkpoint_tag,
        )
        trainer.deepspeed.save_checkpoint(args.training_args.output_dir, tag=checkpoint_tag)  # type: ignore

        if is_main_process():
            out_path = os.path.join(str(best_model_path), "pytorch_model.bin")
            logger.info("converting the best weights to fp32", out_path=out_path)
            from deepspeed.utils.zero_to_fp32 import (
                convert_zero_checkpoint_to_fp32_state_dict,
            )

            convert_zero_checkpoint_to_fp32_state_dict(
                args.training_args.output_dir,
                output_file=out_path,
                tag=checkpoint_tag,
            )


if __name__ == "__main__":
    train_autofix()
