from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from mlx_lm.tokenizer_utils import TokenizerWrapper


def build_tokenizer_data(tokenizer: TokenizerWrapper) -> TokenEnforcerTokenizerData:
    vocab_size = tokenizer.vocab_size
    zero_tokens = tokenizer.encode("0", add_special_tokens=False)
    zero_token = zero_tokens[-1]
    special_ids = set(cast(list[int], tokenizer.all_special_ids))
    regular_tokens: list[tuple[int, str, bool]] = []

    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        decoded_regular = tokenizer.decode([token_id])
        decoded_after_zero = tokenizer.decode([zero_token, token_id])[1:]
        regular_tokens.append(
            (
                token_id,
                decoded_after_zero,
                len(decoded_after_zero) > len(decoded_regular),
            )
        )

    def decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids).rstrip("\ufffd")

    raw_eos_ids = tokenizer.eos_token_ids
    eos_ids = list(raw_eos_ids or [])
    return TokenEnforcerTokenizerData(
        regular_tokens=regular_tokens,
        decoder=decode,
        eos_token_id=eos_ids,
        use_bitmask=False,
        vocab_size=vocab_size,
    )


_TOKENIZER_DATA_CACHE: dict[
    int, tuple[TokenizerWrapper, TokenEnforcerTokenizerData]
] = {}


def _cached_tokenizer_data(
    tokenizer: TokenizerWrapper,
) -> TokenEnforcerTokenizerData:
    identity = id(tokenizer)
    cached = _TOKENIZER_DATA_CACHE.get(identity)
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    built = build_tokenizer_data(tokenizer)
    _TOKENIZER_DATA_CACHE[identity] = (tokenizer, built)
    return built


def make_json_schema_logits_processor(
    tokenizer: TokenizerWrapper,
    response_format: str | dict[str, Any] | None,
    prompt_token_count: int,
) -> Callable[[mx.array, mx.array], mx.array] | None:
    if response_format is None:
        return None
    schema = response_format if isinstance(response_format, dict) else None
    tokenizer_data = _cached_tokenizer_data(tokenizer)
    enforcer = TokenEnforcer(tokenizer_data, JsonSchemaParser(schema))
    raw_eos_ids = tokenizer_data.eos_token_id
    eos_ids = (
        {raw_eos_ids} if isinstance(raw_eos_ids, int) else set(raw_eos_ids)
    )

    def process(tokens: mx.array, logits: mx.array) -> mx.array:
        token_ids = cast(list[int], tokens.tolist())
        # lm-format-enforcer treats token 0 as an opaque prompt sentinel and
        # parses every subsequent token as generated output. mlx-lm supplies
        # the complete prompt here, so passing it through makes prompt text
        # part of the JSON parser state. Keep one prompt token as the sentinel
        # and expose only newly generated tokens to the enforcer.
        generated_sequence = token_ids[max(0, prompt_token_count - 1) :]
        allowed = list(
            enforcer.get_allowed_tokens(generated_sequence).allowed_tokens
        )
        if eos_ids.intersection(allowed):
            # The enforcer can expose EOS for a tokenizer state that still
            # decodes to incomplete JSON (for example, inside an open string).
            # mlx-lm may greedily select that EOS and report a normal stop,
            # leaving clients with a truncated top-level response. The first
            # item is lm-format-enforcer's prompt sentinel; validate only the
            # generated suffix before allowing termination.
            generated_text = tokenizer.decode(generated_sequence[1:])
            try:
                json.loads(generated_text)
            except json.JSONDecodeError:
                allowed = [
                    token_id for token_id in allowed if token_id not in eos_ids
                ]
        indices = mx.array(allowed, dtype=mx.int32)
        mask = mx.zeros((logits.shape[-1],), dtype=mx.bool_).at[indices].add(True)
        return mx.where(mask, logits, mx.array(float("-inf"), dtype=logits.dtype))

    return process
