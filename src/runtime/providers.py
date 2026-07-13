"""Provider-specific LangChain model construction."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import RuntimeConfig


def create_chat_model(config: RuntimeConfig) -> BaseChatModel:
    """Create a provider model with hidden SDK retries disabled."""
    common = {
        "model": config.model.name,
        "temperature": config.model.temperature,
        "timeout": config.timeouts.call_seconds,
        "max_retries": 0,
    }

    if config.model.provider == "openai":
        return ChatOpenAI(
            **common,
            max_completion_tokens=config.model.max_output_tokens,
        )

    return ChatAnthropic(
        **common,
        max_tokens=config.model.max_output_tokens,
    )
