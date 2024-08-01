from typing import Optional

import transformers

from ml_utils.distributed_training import MultiProcessLogger


def convert_to_better_transformer(
    model: transformers.PreTrainedModel, logger: Optional[MultiProcessLogger]
) -> tuple[transformers.PreTrainedModel, bool]:
    try:
        return model.to_bettertransformer(), True
    except Exception as e:
        if logger is not None:
            logger.error(
                "failed converting model to better transformer", exception=str(e)
            )

    return model, False
