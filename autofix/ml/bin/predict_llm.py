import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import simple_parsing
import tqdm

import autofix.ml.lib.data_processor as preprocessor
from autofix.ml.lib.data_schemas import PredictionSchema
from ml_utils.distributed_training import MultiProcessLogger

logger = MultiProcessLogger()


@dataclass(frozen=True)
class LLMInferenceArgs:
    end_point: str
    model_name: str
    model_artifact_id: str
    data_id: str
    num_predictions: int
    temperature: float
    num_shots: int
    max_new_tokens: int

    @staticmethod
    def parse() -> "LLMInferenceArgs":
        return simple_parsing.parse(LLMInferenceArgs)


def predict_llm() -> None:
    args = LLMInferenceArgs.parse()
    data_id = args.data_id

    all_test_data = pd.read_parquet("data/paper_test.parquet")
    all_train_data = pd.read_parquet("data/paper_train.parquet")

    logger.info(
        "Starting inference with settings",
        data_id=data_id,
        model_name=args.model_name,
        num_return_seqs=args.num_predictions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_shots=args.num_shots,
        num_train_samples=len(all_train_data),
        num_test_samples=len(all_test_data),
    )

    checkpoint_dir = f"{args.model_name}_{data_id}_predictions_checkpoint"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpointed_rules = set(os.listdir(checkpoint_dir))

    grouped_by_rule = all_test_data.groupby(PredictionSchema.rule)

    for rule_id, rule_df in grouped_by_rule:
        if f"{rule_id}.parquet" in checkpointed_rules:
            logger.warning("Found cached results. Skipping", rule=rule_id)
            continue

        logger.info("running for", rule=rule_id)
        all_rule_predictions = []

        for _, sample in tqdm.tqdm(rule_df.iterrows(), total=len(rule_df)):
            conversation = preprocessor.create_conversation_with_examples(
                query=sample, few_shot_examples=all_train_data, num_shots=args.num_shots
            )
            inference_not_done = True
            max_tries = 3
            current_tries = 0
            while inference_not_done:
                try:
                    input_data = {
                        "model_input": {
                            "temperature": args.temperature,
                            "num_generations": args.num_predictions,
                            "max_tokens": args.max_new_tokens,
                            "context": "Assistant is a code assistant designed to fix issues in given code snippets.\nInstructions:\n-Do not generate additional text or code. Output only the fixed code snippet\n-Do not generate explanations, comments, notes. Note that the code we provide is incomplete, it is intentionally reduced to a smaller snippet, do not try to complete it in anyway. Leave evertything as it is and just apply the changes related to the fix.",
                            "messages": conversation,
                        },
                        "provider": "openai",
                        "model": args.model_name,
                    }
                    response = requests.post(
                    args.end_point,
                        headers={"Authorization": "Bearer " + os.environ["TOKEN"]},
                        json=input_data,
                    )
                    inference_not_done = response.status_code != 200
                    if inference_not_done:
                        logger.warning(
                            "A non 200 response occured", status=response.status_code
                        )
                        raise RuntimeError("A non 200 response occured")
                except Exception as e:
                    current_tries += 1
                    if current_tries == max_tries:
                        logger.error("Maximum number of tries exceeded")
                        break

                    logger.warning("caught an exception, will try again soon", error=e)
                    time.sleep(60)

            try:
                predictions: list[str] = [
                    message["content"] for message in response.json()["messages"]
                ]
            except Exception as e:
                logger.error("caught an exception in accessing predictions", error=e)
                predictions = [
                    "<FAILED_INFERENCE>" for _ in range(args.num_predictions)
                ]

            all_rule_predictions.append(predictions)  # type ignore
        rule_df["predictions"] = all_rule_predictions
        rule_df.to_parquet(os.path.join(checkpoint_dir, f"{rule_id}.parquet"))

    logger.info("done with generating predictions")
    all_preds_df = pd.read_parquet(checkpoint_dir)
    logger.info("read predictions from checkpoints", num_samples=len(all_preds_df))

    with tempfile.TemporaryDirectory(prefix="autofix_llm_predictions-") as save_dir:
        prediction_file_path = Path(
            os.path.join(save_dir, "predictions", "predictions.parquet")
        )
        prediction_file_path.parent.mkdir(parents=True, exist_ok=True)
        all_preds_df.to_parquet(
            str(prediction_file_path),
            index=False,
        )

    logger.info(
        "Done",
        data_id=data_id,
        model_id=args.model_artifact_id
    )


if __name__ == "__main__":
    predict_llm()
