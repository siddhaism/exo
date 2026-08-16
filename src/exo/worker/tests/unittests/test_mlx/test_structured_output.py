from typing import cast

import mlx.core as mx
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.structured_output import (
    build_tokenizer_data,
    make_json_schema_logits_processor,
)


class _CharacterTokenizer:
    def __init__(self) -> None:
        self.tokens = ["0", "{", "}", '"', ":", ",", "v", "a", "l", "u", "e", "x", " "]
        self.vocab_size = len(self.tokens) + 1
        self.eos_token_id = len(self.tokens)
        self.eos_token_ids = {self.eos_token_id}
        self.all_special_ids = [self.eos_token_id]

    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.tokens.index(character) for character in value]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.tokens[token_id]
            for token_id in token_ids
            if token_id < len(self.tokens)
        )


def test_json_schema_enforcer_accepts_only_schema_valid_sequence() -> None:
    tokenizer = cast(TokenizerWrapper, cast(object, _CharacterTokenizer()))
    tokenizer_data = build_tokenizer_data(tokenizer)
    enforcer = TokenEnforcer(
        tokenizer_data,
        JsonSchemaParser(
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            }
        ),
    )
    prompt_sentinel = 999
    generated = tokenizer.encode('{"value":"x"}', add_special_tokens=False)
    sequence = [prompt_sentinel]

    for token_id in generated:
        allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
        assert token_id in allowed
        sequence.append(token_id)

    allowed_after_object = enforcer.get_allowed_tokens(sequence).allowed_tokens
    assert tokenizer.eos_token_id in allowed_after_object


def test_json_schema_enforcer_honors_string_length_bounds() -> None:
    character_tokenizer = _CharacterTokenizer()
    tokenizer = cast(TokenizerWrapper, cast(object, character_tokenizer))
    enforcer = TokenEnforcer(
        build_tokenizer_data(tokenizer),
        JsonSchemaParser(
            {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "maxLength": 1},
                },
                "required": ["value"],
                "additionalProperties": False,
            }
        ),
    )
    sequence = [999]
    for token_id in tokenizer.encode('{"value":"x', add_special_tokens=False):
        assert token_id in enforcer.get_allowed_tokens(sequence).allowed_tokens
        sequence.append(token_id)

    allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
    assert character_tokenizer.tokens.index('"') in allowed
    assert character_tokenizer.tokens.index("x") not in allowed


def test_logits_processor_excludes_the_complete_prompt_from_json_state() -> None:
    tokenizer = cast(TokenizerWrapper, cast(object, _CharacterTokenizer()))
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    prompt = tokenizer.encode("value", add_special_tokens=False)
    generated = tokenizer.encode('{"value":"x"}', add_special_tokens=False)
    processor = make_json_schema_logits_processor(tokenizer, schema, len(prompt))
    assert processor is not None

    sequence = list(prompt)
    for expected_token in generated:
        logits = mx.zeros((tokenizer.vocab_size,))
        constrained = processor(mx.array(sequence), logits)
        assert bool(mx.isfinite(constrained[expected_token]).item())
        sequence.append(expected_token)

    constrained = processor(
        mx.array(sequence), mx.zeros((tokenizer.vocab_size,))
    )
    assert bool(mx.isfinite(constrained[tokenizer.eos_token_id]).item())


def test_logits_processor_tracks_output_after_manual_prefill_tail() -> None:
    tokenizer = cast(TokenizerWrapper, cast(object, _CharacterTokenizer()))
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    # EXO manually prefills the complete prompt and gives stream_generate only
    # its final two tokens. The processor must use that visible tail length,
    # regardless of how large the original prompt was.
    visible_prompt_tail = tokenizer.encode("ue", add_special_tokens=False)
    generated = tokenizer.encode('{"value":"x"}', add_special_tokens=False)
    processor = make_json_schema_logits_processor(
        tokenizer, schema, len(visible_prompt_tail)
    )
    assert processor is not None

    sequence = list(visible_prompt_tail)
    for index, expected_token in enumerate(generated):
        constrained = processor(
            mx.array(sequence), mx.zeros((tokenizer.vocab_size,))
        )
        assert bool(mx.isfinite(constrained[expected_token]).item())
        sequence.append(expected_token)
        if index < len(generated) - 1:
            constrained = processor(
                mx.array(sequence), mx.zeros((tokenizer.vocab_size,))
            )
            assert not bool(
                mx.isfinite(constrained[tokenizer.eos_token_id]).item()
            )

    constrained = processor(
        mx.array(sequence), mx.zeros((tokenizer.vocab_size,))
    )
    assert bool(mx.isfinite(constrained[tokenizer.eos_token_id]).item())
