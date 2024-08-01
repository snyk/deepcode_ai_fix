import os
from typing import Any

import transformers

from ml_utils.distributed_training import MultiProcessLogger

logger = MultiProcessLogger()


def load_suitable_model(
    model_identifier: str,
    **from_pretrained_kwargs,
) -> Any:  # DO NOT CHANGE THIS TYPE ANNOTATION
    """Loads the model depending on model model_identifier.
    Args:
        model_identifier: either local model directory or an HuggingFace ID
        from_pretrained_kwargs: all the keyword arguments that transformers.from_pretrained accepts can be passed here.
    """

    loaders = [
        transformers.AutoModelForCausalLM,
        transformers.AutoModelForSeq2SeqLM,
    ]
    for loader in loaders:
        try:
            model = loader.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                **from_pretrained_kwargs,
            )
            logger.info(
                "Loaded the model",
                model_id=model_identifier,
                dtype=model.dtype,
                model_type=loader.__name__,
            )
            return model
        except Exception as e:
            logger.warning(
                "Failed to load the model. Will try the next loader",
                failed_loader=loader.__name__,
                error=e,
            )

    raise ValueError("Failed loading the model")


def load_suitable_tokenizer(
    model_identifier: str,
) -> transformers.PreTrainedTokenizerBase:
    """Loads the tokenizer depending on model_identifier.

    Args:
        model_identifier: can be either local model path directory or an HuggingFace ID
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_identifier, trust_remote_code=True
    )
    logger.info(
        "Loaded the tokenizer.",
        path=model_identifier,
        model_type=type(tokenizer),
        num_tokens=len(tokenizer),
    )
    return tokenizer
