import os
from typing import cast

import pandas as pd
import structlog
import torch
import tqdm
import transformers
from datasets.arrow_dataset import Dataset
from pandera.typing import DataFrame
from transformers.pipelines.pt_utils import KeyDataset

import autofix.ml.lib.args as arguments
import autofix.ml.lib.data_processor as data_processor
import autofix.ml.utils.loading as ld
from autofix.ml.lib.data_schemas import LabelledDataSchema
from autofix.ml.lib.data_schemas import PredictionSchema
from autofix.ml_inference.inference_pipeline import make_generation_config
from autofix.ml_inference.inference_pipeline import make_inference_pipeline
from ml_utils.model_type import ModelType

logger = structlog.get_logger()
transformers.logging.set_verbosity_info()


def prediction_worker(
    args: arguments.AutofixArgs,
    data_id: str,
    single_test_data: DataFrame[LabelledDataSchema],
    pipeline: transformers.Pipeline,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> list[str]:
    # Step 1 - preprocess the dataset
    test_ds = Dataset.from_pandas(single_test_data)

    logger.info(
        f"Preprocessing the test dataset",
        num_samples=len(single_test_data),
        data_id=data_id,
    )
    PROMPT_COLUMN_NAME = "prompt"
    model_type = ModelType.infer_from_model(model)

    def process_batch(batch: DataFrame[LabelledDataSchema]) -> dict[str, list[str]]:
        return {
            PROMPT_COLUMN_NAME: data_processor.make_prompts(
                batch, model_type, tokenizer, include_post=False  # type: ignore
            )
        }

    test_ds = test_ds.map(
        process_batch,
        remove_columns=test_ds.column_names,
        keep_in_memory=args.dataset_args.keep_in_memory,
        batched=True,
        batch_size=args.dataset_args.preprocessing_batch_size,
        desc=f"preprocessing the test dataset {data_id}.",
    )

    # Step 2 - generate the model predictions.
    logger.info(
        "Generating predictions...",
        num_samples=len(test_ds),
        maxlen=args.model_args.tokenizer_max_length,
        num_beams=args.inference_args.beam_size,
        batch_size=args.training_args.per_device_eval_batch_size,
    )

    test_ds_wrapper = KeyDataset(
        cast(torch.utils.data.Dataset, test_ds), key=PROMPT_COLUMN_NAME
    )
    pipeline_iterator = pipeline(
        test_ds_wrapper,
        num_workers=args.training_args.dataloader_num_workers,
        batch_size=args.training_args.per_device_eval_batch_size,
        **make_generation_config(args.inference_args, tokenizer),
    )

    # `predictions` is a list (with size `num_samples`) of lists (with size `num_return_seqs`) that contains
    # the beam search predictions for every sample.
    predictions = [
        [out["generated_text"] for out in cast(list[dict], predictions_for_one_sample)]
        # `predictions_for_one_sample` is a list that contains the `num_return_seqs` predictions for a sample.
        for predictions_for_one_sample in tqdm.tqdm(
            pipeline_iterator,
            desc=f"Generating predictions {data_id}.",
        )
    ]

    return predictions  # type: ignore


def predict_autofix() -> None:
    # Step 0 - setup.
    args = arguments.parse_autofix_args_from_cmd()
    if not args.model_args.model_id:
        raise RuntimeError("model_id can not be empty or None.")
    os.makedirs(args.training_args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Step 1 - fetch the data.
    single_test_data = DataFrame[LabelledDataSchema](pd.read_parquet(args.dataset_args.data_id))
    model_path = str(args.model_args.model_id)
    logger.info("Finished fetching data and model")

    # Step 2 - load the model and the tokenizer.
    torch_dtype = torch.float32
    if args.training_args.bf16 or args.training_args.bf16_full_eval:
        torch_dtype = torch.bfloat16
    elif args.training_args.fp16 or args.training_args.fp16_full_eval:
        torch_dtype = torch.float16
    logger.info(
        "Loading the model", torch_dtype=torch_dtype, device=device
    )

    tokenizer = ld.load_suitable_tokenizer(model_path)
    model_loading_kwargs: dict = {
        "torch_dtype": torch_dtype,
        "device_map": str(device),
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

    model = ld.load_suitable_model(
        model_path,  **model_loading_kwargs
    )

    gpu_memory_used = (
        torch.cuda.max_memory_allocated(device=device)
        if torch.cuda.is_available()
        else 0
    )
    logger.info(
        "Done loading model", gpu_memory_used=gpu_memory_used
    )

    USE_TRUNCATION = True
    MAX_INPUT_SIZE = args.model_args.tokenizer_max_length
    logger.info(
        "Creating the inference pipeline",
        truncation=USE_TRUNCATION,
        max_input_size=MAX_INPUT_SIZE,
    )
    pipeline = make_inference_pipeline(
        model,
        tokenizer,
        truncate_input=USE_TRUNCATION,
        max_input_size=MAX_INPUT_SIZE,
        device=None,
    )

    # Step 3 - generate predictions.
    predictions = prediction_worker(
            args, args.dataset_args.data_ids, single_test_data, pipeline, model, tokenizer # type: ignore
    )
    # Step 4 - add the "predictions" column and save the final parquet.
    assert (
        predictions is not None
    ), "Predictions in main process must be returned."
    single_test_data[PredictionSchema.predictions] = predictions
    single_test_data.to_parquet(
        "predictions.parquet",
        index=False,
    )


if __name__ == "__main__":
    predict_autofix()
