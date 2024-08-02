from enum import Enum

import transformers


class ModelType(Enum):
    CAUSAL = "causal"
    SEQ2SEQ = "seq2seq"

    @staticmethod
    def infer_from_model(model: transformers.PreTrainedModel) -> "ModelType":
        t: str = model.config.model_type

        if t in ["t5", "codet5p"]:
            return ModelType.SEQ2SEQ

        if t in [
            "gpt2",
            "gpt_bigcode",
            "mosaic_gpt",
            "llama",
            "mpt",
            "mixtral",
            "stablelm_epoch",
            "starcoder2",
        ]:
            return ModelType.CAUSAL

        raise ValueError(f"unknown model type for the '{t}' architecture")
