from typing import Any, Optional, Union

import torch
import torch.utils.data.dataset
import transformers
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import autofix.ml.lib.data_processor as data_processor
from autofix.ml.lib.args import InferenceArgs
from ml_utils.distributed_training import MultiProcessLogger
from ml_utils.model_type import ModelType

logger = MultiProcessLogger()


_TokenizerT = Union[
    PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
]


class _AutofixCausalInferencePipeline(transformers.TextGenerationPipeline):
    def __init__(
        self,
        truncate_input: bool,
        max_input_size: Optional[int],
        log_metrics: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.truncate_input = truncate_input
        self.max_input_size = max_input_size
        self.log_metrics = log_metrics

    def preprocess(self, inputs, *args, **kwargs):
        # `self.tokenizer` is initialized in the base class constructor.
        assert self.tokenizer is not None

        tokenized_input = data_processor.tokenize_prompts(
            inputs,
            labels=None,
            tokenizer=self.tokenizer,
            truncation=self.truncate_input,
            max_length=self.max_input_size,
            model_type=ModelType.CAUSAL,
            return_tensors="pt",
        )
        # Setting this is required to make the rest of `transformers.TextGenerationPipeline` to work.
        tokenized_input["prompt_text"] = inputs
        return tokenized_input

    def postprocess(
        self,
        model_outputs,
        return_type=transformers.pipelines.text2text_generation.ReturnType.TEXT,
        clean_up_tokenization_spaces=False,
    ):
        return super().postprocess(
            model_outputs, return_type, clean_up_tokenization_spaces  # type: ignore
        )

    def __call__(self, *args, **kwargs):
        # `TextGenerationPipeline` doesn't use `torch.no_grad` by default, so we
        # have to do it ourselves.

        # `return_full_text=True` makes the pipeline return only newly generated text
        # (instead of the input + newly generated text).
        kwargs["return_full_text"] = False
        with torch.no_grad():
            return super().__call__(*args, **kwargs)

class _AutofixSeq2SeqPipeline(transformers.Text2TextGenerationPipeline):
    def __init__(
        self,
        truncate_input: bool,
        max_input_size: Optional[int],
        log_metrics: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.truncate_input = truncate_input
        self.max_input_size = max_input_size
        self.log_metrics = log_metrics

    def preprocess(self, inputs, *args, **kwargs):
        # `self.tokenizer` is initialized in the base class constructor.
        assert self.tokenizer is not None

        tokenized_input = data_processor.tokenize_prompts(
            inputs,
            labels=None,
            tokenizer=self.tokenizer,
            truncation=self.truncate_input,
            max_length=self.max_input_size,
            model_type=ModelType.SEQ2SEQ,
            # the rest of `TextGenerationPipeline` expects tensors, not lists.
            return_tensors="pt",
        )
        return tokenized_input

    def postprocess(
        self,
        model_outputs,
        return_type=transformers.pipelines.text2text_generation.ReturnType.TEXT,
        clean_up_tokenization_spaces=False,
    ):
        return super().postprocess(
            model_outputs, return_type, clean_up_tokenization_spaces
        )

    def __call__(self, *args, **kwargs):
        # `Text2TextGenerationPipeline` doesn't use `torch.no_grad` by default, so we
        # have to do it ourselves.
        with torch.no_grad():
            return super().__call__(*args, **kwargs)

def make_inference_pipeline(
    model: PreTrainedModel,
    tokenizer: _TokenizerT,
    truncate_input: bool,
    max_input_size: int,
    log_metrics: bool = False,
    device: Optional[torch.device] = None,
) -> transformers.Pipeline:
    model_type = ModelType.infer_from_model(model)

    model.eval()
    if model_type == ModelType.SEQ2SEQ:
        return _AutofixSeq2SeqPipeline(
            tokenizer=tokenizer,
            model=model,
            truncate_input=truncate_input,
            max_input_size=max_input_size,
            log_metrics=log_metrics,
        )

    if model_type == ModelType.CAUSAL:
        if tokenizer.padding_side == "right":
            # If we're using a causal (decoder-only) model, it's important to pad on the left, so the input looks like:
            #   <PAD>...<PAD>fix BUG_TYPE<buggy_code>\nfixed:\n<fixed version of code >
            #   <------------ prompt for the model -----------><--- model's output --->
            # and NOT like:
            #   fix BUG_TYPE<buggy_code>\nfixed:\n<PAD>...<PAD><fixed version of code >
            #   <------------ prompt for the model -----------><--- model's output --->
            # The reason for this is that the model is not trained to have the <PAD>...<PAD> gap between
            # the prompt and the output, so having it during inference may severely lower the output quality.
            # Also note that for training, the padding side doesn't really matter, because the prompt and the answer
            # are supplied as one contiguous piece of text without any padding tokens in between.
            # For more details, see: https://github.com/huggingface/transformers/issues/3021#issuecomment-1454266627
            logger.warning(
                "switching the tokenizer padding side from 'right' to 'left' for a causal LM"
            )
            tokenizer.padding_side = "left"
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        return _AutofixCausalInferencePipeline(
            tokenizer=tokenizer,
            model=model,
            truncate_input=truncate_input,
            max_input_size=max_input_size,
            log_metrics=log_metrics,
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported ModelType: {model_type}")


def make_generation_config(
    inference_args: InferenceArgs, tokenizer: _TokenizerT
) -> dict[str, Any]:
    if inference_args.max_new_tokens is None:
        raise ValueError(
            "please set the --max_new_tokens argument to limit the max output size"
        )

    eos_token_ids = [tokenizer.eos_token_id]
    if tokenizer.chat_template is None:
        # Sanity check and extend eos_token_ids
        end_of_fixed_code_str: str = data_processor._END_FIXED_CODE
        end_of_fixed_code_token_id: int = tokenizer.convert_tokens_to_ids(end_of_fixed_code_str)  # type: ignore

        if end_of_fixed_code_str != tokenizer.convert_ids_to_tokens(end_of_fixed_code_token_id):  # type: ignore
            # A sanity check that `end_of_fixed_code_str` is used as an independent token in the tokenizer.
            raise Exception(
                f"the '{end_of_fixed_code_str}' is not present in the tokenizer's vocabulary (the tokenizer returned token_id={end_of_fixed_code_token_id} instead"
            )
        eos_token_ids = [tokenizer.eos_token_id, end_of_fixed_code_token_id]

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return {
        "num_return_sequences": inference_args.num_return_seqs,
        "early_stopping": inference_args.early_stopping,
        # We set the `max_length` to None explicitly to avoid the noisy HF warnings.
        "max_length": None,
        "max_new_tokens": inference_args.max_new_tokens,
        "num_beams": inference_args.beam_size,
        # Setting the padding token id is needed to silence the warnings that HF throws at every call to `predict`.
        "pad_token_id": tokenizer.pad_token_id,
        # We use `end_of_fixed_code_token_id` as an "end of sequence" token to stop predicting at `end_of_fixed_code_token_id`.
        "eos_token_id": eos_token_ids,
    }
