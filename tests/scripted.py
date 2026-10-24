"""A chat model that replays scripted replies, so the agent runs without the API."""

import itertools
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = itertools.count(1)


def calls(*tool_calls: tuple[str, dict[str, Any]], text: str = "") -> AIMessage:
    """An assistant message that calls the given tools."""
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}
            for name, args in tool_calls
        ],
    )


def answer(text: str, **metadata: Any) -> AIMessage:
    return AIMessage(content=text, response_metadata=metadata)


class ScriptedChatModel(BaseChatModel):
    replies: list[AIMessage]
    received: list[list[BaseMessage]] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        self.tool_names = [getattr(tool, "name", None) or tool["name"] for tool in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        if not self.replies:
            raise AssertionError("the agent called the model more often than scripted")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])
