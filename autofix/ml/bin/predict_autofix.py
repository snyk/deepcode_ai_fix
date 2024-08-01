import contextlib
import multiprocessing.synchronize
import os
from typing import Optional, cast

import pandas as pd
import structlog
import torch
import tqdm
import transformers
from datasets.arrow_dataset import Dataset
from pandera.typing import DataFrame
from transformers.pipelines.pt_utils import KeyDataset

import autofix.ml.lib.args as arguments
import autofix.ml.lib.constants as const
import autofix.ml.lib.data_processor as data_processor
import autofix.ml.utils.loading as ld
from autofix.ml.lib.data_schemas import LabelledDataSchema
from autofix.ml.lib.data_schemas import PredictionSchema
from autofix.ml.utils.optimizations import convert_to_better_transformer
from autofix.ml_inference.inference_pipeline import make_generation_config
from autofix.ml_inference.inference_pipeline import make_inference_pipeline
from ml_utils.model_type import ModelType

logger = structlog.get_logger()
transformers.logging.set_verbosity_info()


def num_workers() -> int:
    if os.environ.get("NO_GPU") is not None:
        return 1
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        return num_gpus
    # If no GPUs are available, train with one process on CPU.
    return 1


# `barrier` must be a global variable to use it across different processes.
barrier = multiprocessing.Barrier(num_workers())


def is_multi_processing() -> bool:
    return num_workers() > 1


def is_main_process(worker_id: int) -> int:
    return worker_id == 0


def optimize_model(
    model: transformers.PreTrainedModel, args: arguments.AutofixArgs
) -> transformers.PreTrainedModel:
    # `convert_to_better_transformer` is actually failing for some models (e.g. CodeGen).
    # We enable it to keep our serving and evaluation code aligned. Our serving uses BetterTransformer because it is much faster and we must evaluate what we serve and serve what we evaluate
    model, success = convert_to_better_transformer(model, logger=logger)
    if success:
        logger.info("Converted to BetterTransformer")
        if args.training_args.per_device_eval_batch_size != 1:
            logger.warning(
                "Setting per_device_eval_batch_size to 1 because BetterTransformer does not support batched inference. "
            )
            args.training_args.per_device_eval_batch_size = 1

    if (
        args.training_args.torch_compile
        or args.training_args.torch_compile_mode is not None
    ):
        model = cast(
            transformers.PreTrainedModel,
            torch.compile(
                model,
                mode=args.training_args.torch_compile_mode,
            ),
        )
        logger.info("Torch compiled the model.")

    return model


@contextlib.contextmanager
def main_worker_first(worker_id: int, barrier: multiprocessing.synchronize.Barrier):
    """A context manager to execute some code on the first worker first and only then on all other workers"""
    if is_main_process(worker_id):
        yield
        barrier.wait()
    else:
        barrier.wait()
        yield


def prediction_worker(
    worker_id: int,
    args: arguments.AutofixArgs,
    data_id: str,
    single_test_data: DataFrame[LabelledDataSchema],
    pipeline: transformers.Pipeline,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> Optional[DataFrame]:
    # Step 1 - preprocess the dataset
    test_ds = Dataset.from_pandas(single_test_data)

    logger.info(
        f"Sharding the dataset",
        num_shards=num_workers(),
        data_id=data_id,
        worker_id=worker_id,
    )
    test_ds = test_ds.shard(
        num_shards=num_workers(),
        index=worker_id,
        contiguous=True,
        keep_in_memory=args.dataset_args.keep_in_memory,
    )

    logger.info(
        f"Preprocessing the test dataset",
        num_samples=len(single_test_data),
        data_id=data_id,
        worker_id=worker_id,
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
        desc=f"preprocessing the test dataset {data_id} for shard {worker_id}",
    )

    # Step 2 - generate the model predictions.
    logger.info(
        "Generating predictions...",
        num_samples=len(test_ds),
        maxlen=args.model_args.tokenizer_max_length,
        num_beams=args.inference_args.beam_size,
        worker_id=worker_id,
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

    # Wait to make all workers start predicting at the same time.
    # Not strictly necessary per se, but makes the logs nicer.
    barrier.wait()

    # `predictions` is a list (with size `num_samples`) of lists (with size `num_return_seqs`) that contains
    # the beam search predictions for every sample.
    predictions = [
        [out["generated_text"] for out in cast(list[dict], predictions_for_one_sample)]
        # `predictions_for_one_sample` is a list that contains the `num_return_seqs` predictions for a sample.
        for predictions_for_one_sample in tqdm.tqdm(
            pipeline_iterator,
            desc=f"Generating predictions {data_id} for shard: {worker_id}",
            position=worker_id,
        )
    ]

    barrier.wait()

    # Step 3 - if multiprocessing, save the shards to a temp folder and bring them back together in the main process.
    if not is_multi_processing():
        return predictions # type: ignore

    def shard_path(shard_id: int) -> str:
        return os.path.join(
            args.training_args.output_dir,
            f"predictions_{data_id}_shard_{shard_id}.parquet",
        )

    out_path = shard_path(worker_id)
    if os.path.exists(out_path):
        os.remove(out_path)
    logger.info(
        f"Saving generated predictions",
        data_id=data_id,
        worker_id=worker_id,
        shard_path=out_path,
        num_predictions=len(predictions),
    )
    PREDICTIONS_COLUMN_NAME = "predictions"
    df = pd.DataFrame({PREDICTIONS_COLUMN_NAME: predictions})
    df.to_parquet(out_path, compression="snappy")

    logger.info(
        "Process done writing its shard",
        worker_id=worker_id,
    )
    barrier.wait()

    if not is_main_process(worker_id):
        return None
    # Only main process returns merged shards.

    logger.info(
        "reading the shards in the main process",
        num_shards=num_workers(),
    )
    shards = [
        pd.read_parquet(shard_path(shard_id)) for shard_id in range(0, num_workers())
    ]
    return pd.concat(shards, ignore_index=True)[PREDICTIONS_COLUMN_NAME]  # type: ignore


def main_worker(worker_id: int) -> None:
    # Step 0 - setup.
    args = arguments.parse_autofix_args_from_cmd()
    if not args.model_args.model_id:
        raise RuntimeError("model_id can not be empty or None.")
    os.makedirs(args.training_args.output_dir, exist_ok=True)

    output_arts: list = []

    device = (
        torch.device(f"cuda:{worker_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Step 1 - fetch the data.
    with main_worker_first(worker_id, barrier):
        test_datas: list[DataFrame[LabelledDataSchema]] = []
        # for data_id, data_tag in zip(
        #     args.dataset_args.data_ids, args.dataset_args.data_tags
        # ):
        single_test_data = DataFrame[LabelledDataSchema](pd.read_parquet("test.parquet"))
        test_datas.append(single_test_data) # type: ignore
        model_path = str(args.model_args.model_id)
        logger.info("Finished fetching data and model", worker_id=worker_id)

    # Step 2 - load the model and the tokenizer.
    torch_dtype = torch.float32
    if args.training_args.bf16 or args.training_args.bf16_full_eval:
        torch_dtype = torch.bfloat16
    elif args.training_args.fp16 or args.training_args.fp16_full_eval:
        torch_dtype = torch.float16
    logger.info(
        "Loading the model", torch_dtype=torch_dtype, worker_id=worker_id, device=device
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
    # if isinstance(model, transformers.PreTrainedModel):
    #     model = optimize_model(model, args)
    gpu_memory_used = (
        torch.cuda.max_memory_allocated(device=device)
        if torch.cuda.is_available()
        else 0
    )
    logger.info(
        "Done loading model", worker_id=worker_id, gpu_memory_used=gpu_memory_used
    )

    USE_TRUNCATION = True
    MAX_INPUT_SIZE = args.model_args.tokenizer_max_length
    logger.info(
        "Creating the inference pipeline",
        truncation=USE_TRUNCATION,
        max_input_size=MAX_INPUT_SIZE,
        worker_id=worker_id,
    )
    pipeline = make_inference_pipeline(
        model,
        tokenizer,
        truncate_input=USE_TRUNCATION,
        max_input_size=MAX_INPUT_SIZE,
        device=None,
    )

    for data_id, single_test_data, output_art in zip(
        args.dataset_args.data_ids, test_datas, output_arts
    ):
        # Step 3 - generate predictions.
        predictions = prediction_worker(
            worker_id, args, data_id, single_test_data, pipeline, model, tokenizer
        )
        # Step 4 - add the "predictions" column and save the final parquet.
        if is_main_process(worker_id):
            assert (
                predictions is not None
            ), "Predictions in main process must be returned."
            single_test_data[PredictionSchema.predictions] = predictions
            predictions_file_path = output_art.get_predictions_file()
            predictions_file_path.parent.mkdir(parents=True, exist_ok=True)
            single_test_data.to_parquet(
                str(predictions_file_path),
                compression=const.PARQUET_COMPRESSION,
                index=False,
            )

        # Wait for all workers to finish processing one single_test_data before the next iteration.
        # Not strictly necessary per se, but makes the logs nicer.
        barrier.wait()



def predict_autofix() -> None:
    if not is_multi_processing():
        main_worker(0)
    else:
        logger.info("Spawning processes", num_workers=num_workers())
        torch.multiprocessing.start_processes(
            main_worker,
            nprocs=num_workers(),
            join=True,
            daemon=False,
            start_method="fork",
        )


if __name__ == "__main__":
    predict_autofix()
