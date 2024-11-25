from dataclasses import dataclass
from dataclasses import field
from typing import cast

import transformers
from transformers.hf_argparser import DataClassType


@dataclass
class DatasetArgs:
    dataset_seed: int = field(
        default=42, metadata={"help": "The seed used for all random operations."}
    )
    train_size: float = field(
        default=0.9,
        metadata={"help": "Relative train split size as float in range (0,1)"},
    )
    data_id: str = field(
        default="non-existent",
        metadata={"help": "Repeatable data id of the artifact."},
    )
    preprocessing_batch_size: int = field(
        default=1,
        metadata={
            "help": "The batch size to use during dataset preprocessing. For more details see: https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.map.batch_size"
        },
    )
    keep_in_memory: bool = field(
        default=True,
        metadata={
            "help": "Whether the `dataset.map` results should be kept in memory or written to a HuggingFace cache file. For more details, see: https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.map.keep_in_memory"
        },
    )

@dataclass
class ModelArgs:
    model_name: str = field(
        metadata={"help": "Model name to be used, e.g. `t5-large`."}
    )
    tokenizer_max_length: int = field(
        metadata={"help": "Maximum length for tokenized samples"}
    )
    model_id: str = field(
        metadata={
            "help": (
                "The model identifier"
            )
        },
    )
    use_bits_and_bytes: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bits and bytes config to load and fine-tune the model."
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use parameter efficient fine-tuning. If use_bits_and_bytes is True, this must be se to True as well."
            )
        },
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. Needs batch size of 1 to work properly."
            )
        },
    )

@dataclass
class InferenceArgs:
    max_new_tokens: int = field(
        metadata={
            "help": "The number of new tokens to generate. It must be set to limit the maximum number of tokens generated by the model."
        }
    )
    beam_size: int = field(
        default=1, metadata={"help": "Beam size during validation and prediction"}
    )
    num_return_seqs: int = field(
        default=1,
        metadata={
            "help": "The number of returned sequences for each sample during prediction"
        },
    )
    early_stopping: bool = field(
        default=False,
        metadata={
            "help": "Whether to stop generation once <num_return_seqs> many sequences are done during beam search or continue with it untill all sequences are finished."
        },
    )


@dataclass
class AutofixArgs:
    model_args: ModelArgs
    dataset_args: DatasetArgs
    training_args: transformers.TrainingArguments
    inference_args: InferenceArgs


def parse_autofix_args_from_cmd() -> AutofixArgs:
    argument_types = [
        ModelArgs,
        DatasetArgs,
        transformers.TrainingArguments,
        InferenceArgs,
    ]
    dataclass_types = [cast(DataClassType, arg_type) for arg_type in argument_types]
    hf_parser = transformers.HfArgumentParser(dataclass_types=dataclass_types)

    ma: ModelArgs
    da: DatasetArgs
    ta: transformers.TrainingArguments
    ia: InferenceArgs
    ma, da, ta, ia = hf_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[: len(argument_types)]
    return AutofixArgs(
        model_args=ma,
        dataset_args=da,
        training_args=ta,
        inference_args=ia,
    )
