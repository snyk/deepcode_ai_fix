from typing import List, Optional, TypeVar, Union

import pandas as pd
import tokenizers
from pandera.typing import DataFrame
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from autofix.ml.lib.data_schemas import LabelledDataSchema
from ml_utils.model_type import ModelType

TDataFrame = TypeVar("TDataFrame", bound=pd.DataFrame)

# These are special tokens that will be added to the tokenizer.
_BEGIN_BUGGY_CODE = "<beginbuggycode>"
_END_BUGGY_CODE = "<endbuggycode>"
_BEGIN_FIXED_CODE = "<beginfixedcode>"
_END_FIXED_CODE = "<endfixedcode>"


def make_special_tokens() -> list[tokenizers.AddedToken]:
    return [
        tokenizers.AddedToken(_BEGIN_BUGGY_CODE, normalized=False),
        tokenizers.AddedToken(_END_BUGGY_CODE, normalized=False),
        tokenizers.AddedToken(_BEGIN_FIXED_CODE, normalized=False),
        tokenizers.AddedToken(_END_FIXED_CODE, normalized=False),
    ]

def make_prompt_for_seq2seq_model(
    rule_key: str,
    pre_code: str,
) -> str:
    return f"fix {rule_key}\n{_BEGIN_BUGGY_CODE}\n{pre_code}\n{_END_BUGGY_CODE}\n{_BEGIN_FIXED_CODE}\n"


def wrap_fixed_code_for_seq2seq_model(post_code: str) -> str:
    return f"{post_code}{_END_FIXED_CODE}"

def make_prompt_for_starcoder(
    tokenizer: PreTrainedTokenizer,
    rule_key: str,
    rule_message: str,
    pre_code: str,
    post_code: Optional[str] = None,
) -> str:
    prompt = f"Fix {rule_key}\n\n{rule_message}\n{_BEGIN_BUGGY_CODE}\n{pre_code}\n{_END_BUGGY_CODE}\n{_BEGIN_FIXED_CODE}\n"
    if post_code is not None:
        prompt += f"{post_code}{_END_FIXED_CODE}"
    return prompt


def make_prompt_for_chat(
    tokenizer: PreTrainedTokenizer,
    rule_key: str,
    rule_message: str,
    pre_code: str,
    post_code: Optional[str] = None,
) -> str:
    chat = [
        {
            "role": "user",
            "content": f"Fix the bug {rule_key}: {rule_message} in the code {pre_code}",
        },
    ]
    if post_code is not None:
        chat.append({"role": "assistant", "content": f"{post_code}"})
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )  # type: ignore


def make_prompt_for_causal_model(
    rule_key: str,
    rule_message: str,
    pre_code: str,
    tokenizer: PreTrainedTokenizer,
    post_code: Optional[str] = None,
) -> str:
    if tokenizer.chat_template is not None:
        return make_prompt_for_chat(
            tokenizer=tokenizer,
            rule_key=rule_key,
            rule_message=rule_message,
            pre_code=pre_code,
            post_code=post_code,
        )
    else:
        return make_prompt_for_starcoder(
            tokenizer=tokenizer,
            rule_key=rule_key,
            rule_message=rule_message,
            pre_code=pre_code,
            post_code=post_code,
        )


def make_prompt_for_llm_user(rule_key: str, rule_message: str, pre_code: str) -> str:
    p = f"generate the fixed code for the bug {rule_key} with the error message {rule_message}\n{pre_code}\n"
    return p


def create_conversation_with_examples(
    query, few_shot_examples: DataFrame[LabelledDataSchema], num_shots: int
) -> list[dict[str, str]]:
    shots = few_shot_examples[
        few_shot_examples[LabelledDataSchema.rule] == query[LabelledDataSchema.rule]
    ]
    shots = shots.sample(n=num_shots, random_state=42)

    example_conversation = []
    for _, example in shots.iterrows():
        shot_messages = [
            {
                "role": "USER",
                "content": make_prompt_for_llm_user(
                    example[LabelledDataSchema.rule],
                    example[LabelledDataSchema.message],
                    example[LabelledDataSchema.pre_reduced],
                ),
            },
            {
                "role": "ASSISTANT",
                "content": f"{example[LabelledDataSchema.post_reduced]}\n",
            },
        ]
        example_conversation += shot_messages

    return example_conversation + [
        {
            "role": "USER",
            "content": make_prompt_for_llm_user(
                query[LabelledDataSchema.rule],
                query[LabelledDataSchema.message],
                query[LabelledDataSchema.pre_reduced],
            ),
        }
    ]


def make_prompts(
    batch: DataFrame[LabelledDataSchema],
    model_type: ModelType,
    tokenizer: PreTrainedTokenizer,
    include_post: bool,
) -> List[str]:
    if model_type == ModelType.SEQ2SEQ:
        return [
            make_prompt_for_seq2seq_model(rule_key, pre_code)
            for rule_key, pre_code in zip(
                batch[LabelledDataSchema.rule],
                batch[LabelledDataSchema.pre_reduced],
            )
        ]
    if model_type == ModelType.CAUSAL:
        return [
            make_prompt_for_causal_model(
                rule_key=rule_key,
                rule_message=rule_message,
                pre_code=pre_code,
                tokenizer=tokenizer,
                post_code=post_code if include_post else None,
            )
            for rule_key, rule_message, pre_code, post_code in zip(
                batch[LabelledDataSchema.rule],
                batch[LabelledDataSchema.message],
                batch[LabelledDataSchema.pre_reduced],
                batch[LabelledDataSchema.post_reduced],
            )
        ]
    else:
        raise ValueError(f"{model_type} is not yet handled!")


def tokenize_prompts(
    prompts: Union[str, List[str]],
    labels: Optional[Union[str, List[str]]],
    tokenizer: PreTrainedTokenizer,
    truncation: bool,
    max_length: Optional[int],
    model_type: ModelType,
    return_tensors: Optional[str] = None,
) -> BatchEncoding:
    if model_type == ModelType.SEQ2SEQ:
        if labels is not None:
            if isinstance(labels, str):
                labels = wrap_fixed_code_for_seq2seq_model(labels)
            else:
                labels = [wrap_fixed_code_for_seq2seq_model(label) for label in labels]

        return tokenizer(
            text=prompts,
            text_target=labels,  # type: ignore
            truncation=truncation,
            max_length=max_length,
            padding=False,
            return_tensors=return_tensors,
        )
    if model_type == ModelType.CAUSAL:
        return tokenizer(
            text=prompts,
            truncation=truncation,
            max_length=max_length,
            padding=False,
            return_tensors=return_tensors,
        )
    else:
        raise ValueError(f"{model_type} is not yet handled!")
