from typing import Union, cast

import torch
import torch.utils.data.dataset
import transformers
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from autofix.ml.lib.args import AutofixArgs
from ml_utils.distributed_training import MultiProcessLogger
from ml_utils.model_type import ModelType

logger = MultiProcessLogger()


_TokenizerT = Union[
    PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
]


class AutofixTrainer(transformers.Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: _TokenizerT,
        args: AutofixArgs,
        train_ds: Dataset,
        val_ds: Dataset,
    ):
        model_type = ModelType.infer_from_model(model)
        if model_type == ModelType.CAUSAL:
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
        elif model_type == ModelType.SEQ2SEQ:
            data_collator = transformers.DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding="longest",
                pad_to_multiple_of=8,
                max_length=args.model_args.tokenizer_max_length,
                return_tensors="pt",
            )
        else:
            raise ValueError(f"unsupported ModelType: {model_type}")

        super().__init__(
            model=model,
            train_dataset=cast(torch.utils.data.dataset.Dataset, train_ds),
            eval_dataset=cast(torch.utils.data.dataset.Dataset, val_ds),
            tokenizer=tokenizer,
            args=args.training_args,
            data_collator=data_collator,
        )
