from unittest.mock import MagicMock

import tokenizers
from transformers import PreTrainedTokenizerBase

from autofix.ml.lib import data_processor
from ml_utils import model_type


class TestStarCoderPrompt:
    """Test that `make_prompt_for_starcoder()` produces the expected output."""

    def test_inference(self) -> None:
        """Test when `post_code` is not provided, for inference."""
        mocked_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mocked_tokenizer.eos_token, mocked_tokenizer.eos_token_id = "<|endoftext|>", 0
        prompt = data_processor.make_prompt_for_starcoder(
            mocked_tokenizer,
            rule_key="ArrayConstructor",
            rule_message="Use an array literal instead of the Array constructor.",
            pre_code="var x = new Array();",
        )
        assert prompt == (
            "Fix ArrayConstructor\n\n"
            "Use an array literal instead of the Array constructor.\n"
            "<beginbuggycode>\n"
            "var x = new Array();\n"
            "<endbuggycode>\n"
            "<beginfixedcode>\n"
        )

    def test_training(self) -> None:
        """Test when `post_code` is provided for training."""
        mocked_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mocked_tokenizer.eos_token, mocked_tokenizer.eos_token_id = "<|endoftext|>", 0
        prompt = data_processor.make_prompt_for_starcoder(
            mocked_tokenizer,
            rule_key="ArrayConstructor",
            rule_message="Use an array literal instead of the Array constructor.",
            pre_code="var x = new Array();",
            post_code="var x = [];",
        )
        assert prompt == (
            "Fix ArrayConstructor\n\n"
            "Use an array literal instead of the Array constructor.\n"
            "<beginbuggycode>\n"
            "var x = new Array();\n"
            "<endbuggycode>\n"
            "<beginfixedcode>\n"
            "var x = [];<endfixedcode>"
        )


class TestChatPrompt:
    """Test that `make_prompt_for_chat()` produces the expected output."""

    @staticmethod
    def mock_apply_chat_template(chat, *args, **kwargs):
        return chat

    def test_inference(self) -> None:
        """Test when `post_code` is not provided, for inference"""
        mocked_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mocked_tokenizer.apply_chat_template = self.mock_apply_chat_template
        prompt = data_processor.make_prompt_for_chat(
            mocked_tokenizer,
            rule_key="ArrayConstructor",
            rule_message="Use an array literal instead of the Array constructor.",
            pre_code="var x = new Array();",
        )
        assert prompt == [
            {
                "role": "user",
                "content": "Fix the bug ArrayConstructor: Use an array literal instead of the Array constructor. in the code var x = new Array();",
            }
        ]

    def test_training(self) -> None:
        """Test when `post_code` is provided, for training"""
        mocked_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        mocked_tokenizer.apply_chat_template = self.mock_apply_chat_template
        prompt = data_processor.make_prompt_for_chat(
            mocked_tokenizer,
            rule_key="ArrayConstructor",
            rule_message="Use an array literal instead of the Array constructor.",
            pre_code="var x = new Array();",
            post_code="var x = [];",
        )
        assert prompt == [
            {
                "role": "user",
                "content": "Fix the bug ArrayConstructor: Use an array literal instead of the Array constructor. in the code var x = new Array();",
            },
            {
                "role": "assistant",
                "content": "var x = [];",
            },
        ]


def test_make_special_tokens() -> None:
    special_tokens = data_processor.make_special_tokens()
    special_tokens = set(special_tokens)

    expected_special_tokens = {
        tokenizers.AddedToken("<beginbuggycode>", normalized=False),
        tokenizers.AddedToken("<endbuggycode>", normalized=False),
        tokenizers.AddedToken("<beginfixedcode>", normalized=False),
        tokenizers.AddedToken("<endfixedcode>", normalized=False),
    }
    assert special_tokens == expected_special_tokens


def test_make_prompts_for_few_shot_learning() -> None:
    pre_code = "var x = 11"
    rule_key = "PT"
    rule_message = "Unsanitized input flows into sink"

    llm_prompt = data_processor.make_prompt_for_llm_user(
        rule_key=rule_key, rule_message=rule_message, pre_code=pre_code
    )
    assert (
        llm_prompt
        == "generate the fixed code for the bug PT with the error message Unsanitized input flows into sink\nvar x = 11\n"
    )


def test_create_conversation_with_examples(autofix_df) -> None:
    query = autofix_df.iloc[0]
    few_shot_examples = autofix_df.iloc[1:]

    zero_shot_conversation = data_processor.create_conversation_with_examples(
        query=query, few_shot_examples=few_shot_examples, num_shots=0
    )

    assert zero_shot_conversation == [
        {
            "role": "USER",
            "content": "generate the fixed code for the bug ArrayConstructor with the error message message_array_constructor\nvar x = new Array();\n",
        },
    ]

    one_shot_conversation = data_processor.create_conversation_with_examples(
        query=query, few_shot_examples=few_shot_examples, num_shots=1
    )
    assert one_shot_conversation == [
        {
            "role": "USER",
            "content": "generate the fixed code for the bug ArrayConstructor with the error message message_array_constructor\nvar x = new Array();\n",
        },
        {
            "role": "ASSISTANT",
            "content": "var x = [];\n",
        },
        {
            "role": "USER",
            "content": "generate the fixed code for the bug ArrayConstructor with the error message message_array_constructor\nvar x = new Array();\n",
        },
    ]

    two_shot_conversation = data_processor.create_conversation_with_examples(
        query=query, few_shot_examples=few_shot_examples, num_shots=2
    )
    assert two_shot_conversation == [
        {
            "role": "USER",
            "content": "generate the fixed code for the bug ArrayConstructor with the error message message_array_constructor\nvar x = new Array();\n",
        },
        {
            "role": "ASSISTANT",
            "content": "var x = [];\n",
        },
        {
            "role": "USER",
            "content": "generate the fixed code for the bug ArrayConstructor with the error message message_array_constructor\nvar x = new Array();\n",
        },
        {
            "role": "ASSISTANT",
            "content": "var y = [];\n",
        },
        {
            "role": "USER",
            "content": "generate the fixed code for the bug ArrayConstructor with the error message message_array_constructor\nvar x = new Array();\n",
        },
    ]


def test_tokenizer_prompts(autofix_df) -> None:
    # We omit downloading the tokenizer in unit tests and mock it.
    mocked_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    mocked_tokenizer.chat_template = None
    prompts = data_processor.make_prompts(
        autofix_df,
        model_type.ModelType.CAUSAL,
        tokenizer=mocked_tokenizer,
        include_post=False,
    )

    def tokenize_prompts_and_return_tokenizer_arguments(
        model_type: model_type.ModelType,
    ):
        data_processor.tokenize_prompts(
            prompts=prompts,
            tokenizer=mocked_tokenizer,
            truncation=True,
            max_length=512,
            model_type=model_type,
        )
        call_args = mocked_tokenizer.call_args
        _, kwargs = call_args[0], call_args[1]
        return kwargs

    # For causal models, we do not have a separate field for labels as they are decoder only. The targets are in the prompt already.
    kwargs = tokenize_prompts_and_return_tokenizer_arguments(
        model_type.ModelType.CAUSAL
    )
    assert "text_target" not in kwargs
