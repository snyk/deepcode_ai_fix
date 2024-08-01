from unittest.mock import MagicMock

import transformers

import autofix.ml_inference.inference_pipeline as ip


def test_make_inference_pipeline() -> None:
    model = MagicMock(spec=transformers.PreTrainedModel)
    tokenizer = MagicMock(spec=transformers.PreTrainedTokenizerBase)
    tokenizer.padding_side = "right"
    config = MagicMock(spec=transformers.PretrainedConfig)
    generation_config = MagicMock(spec=transformers.GenerationConfig)
    generation_config.pad_token_id = None
    config.task_specific_params = {}
    config.prefix = ""
    model.config = config
    model.generation_config = generation_config
    model.config.model_type = "llama"  # type: ignore
    pipeline = ip.make_inference_pipeline(
        model,
        tokenizer,
        truncate_input=True,
        max_input_size=512,
    )
    assert isinstance(pipeline, ip._AutofixCausalInferencePipeline)
    assert tokenizer.padding_side == "left"
