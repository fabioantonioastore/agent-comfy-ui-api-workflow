from langchain_anthropic import ChatAnthropic

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
# The API re-runs a declined request on another model; only these models accept it.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)


def build_chat_model(
    model: str = DEFAULT_MODEL,
    *,
    api_key: str,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = 16000,
    refusal_fallback: bool = True,
) -> ChatAnthropic:
    # No temperature, top_p or top_k: current models reject them.
    options = {}
    if refusal_fallback and model in _FALLBACK_MODELS:
        options = {"betas": [_FALLBACK_BETA], "model_kwargs": {"fallbacks": "default"}}
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        effort=effort,
        max_retries=3,
        default_request_timeout=600.0,
        **options,
    )
