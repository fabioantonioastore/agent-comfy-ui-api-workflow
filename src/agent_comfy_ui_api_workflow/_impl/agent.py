import asyncio
import copy
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anthropic
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from .catalog import NodeCatalog
from .env import load_settings
from .families import family_nodes, find_models
from .files import FILE_INPUTS, InputFile, vision_block
from .inventory import ServerInventory
from .middleware import ValidationGate
from .model import DEFAULT_EFFORT, DEFAULT_MODEL, build_chat_model
from .pipelines import AgentPipeline, available_pipelines
from .prompts import system_prompt
from .session import Question, WorkflowDiff, WorkflowSession, diff_workflows
from .tools import build_tools
from .validation import ValidationReport

logger = logging.getLogger(__name__)

Status = Literal["completed", "needs_input", "invalid", "refused", "error"]

# Nodes that load a file from a server folder: with the folder empty they cannot work.
_FOLDER_NODES: dict[str, str] = {
    "CheckpointLoaderSimple": "checkpoints",
    "UNETLoader": "diffusion_models",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "VAELoader": "vae",
    "LoraLoader": "loras",
    "LoraLoaderModelOnly": "loras",
    "UpscaleModelLoader": "upscale_models",
    "ImageUpscaleWithModel": "upscale_models",
    "AudioEncoderLoader": "audio_encoders",
}


def hidden_nodes(inventory: ServerInventory) -> set[str]:
    return {
        node
        for node, folder in _FOLDER_NODES.items()
        if inventory.knows(folder) and not inventory.folder(folder)
    }


@dataclass(frozen=True)
class AgentReply:
    """What a turn produced: the answer, and the workflow as it is now."""

    status: Status
    message: str
    workflow: dict[str, Any]
    question: Question | None = None
    changes: WorkflowDiff = field(default_factory=WorkflowDiff)
    validation: ValidationReport = field(default_factory=ValidationReport)
    # The user files the workflow loads; they go to the server with it.
    input_files: tuple[str, ...] = ()
    # Tokens the turn used, added over its model calls.
    usage: dict[str, int] = field(default_factory=dict)


def _referenced_files(workflow: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for node in workflow.values():
        for name, value in node["inputs"].items():
            if (node["class_type"], name) in FILE_INPUTS and isinstance(value, str) and value not in names:
                names.append(value)
    return tuple(names)


def _usage(messages: list[BaseMessage]) -> dict[str, int]:
    totals = dict.fromkeys(
        ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "model_calls"), 0
    )
    for message in messages:
        if not isinstance(message, AIMessage) or message.usage_metadata is None:
            continue
        usage = message.usage_metadata
        details = usage.get("input_token_details") or {}
        totals["input_tokens"] += usage.get("input_tokens", 0)
        totals["output_tokens"] += usage.get("output_tokens", 0)
        totals["cache_read_tokens"] += details.get("cache_read") or 0
        totals["cache_write_tokens"] += details.get("cache_creation") or 0
        totals["model_calls"] += 1
    return totals


def _final_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.text.strip():
            return message.text.strip()
        if isinstance(message, HumanMessage):
            break
    return ""


class WorkflowAgent:
    """Builds and edits a ComfyUI API workflow, turn by turn, from what the user asks."""

    def __init__(
        self,
        inventory: ServerInventory,
        *,
        workflow: Mapping[str, Any] | None = None,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        api_key: str | None = None,
        env_file: str | Path = ".env",
        chat_model: BaseChatModel | None = None,
        pipelines: Iterable[type[AgentPipeline]] | None = None,
        refusal_fallback: bool = True,
        vision: bool = True,
        max_model_calls: int = 40,
    ) -> None:
        models = find_models(inventory)
        catalog = NodeCatalog(extra=family_nodes(models), hidden=hidden_nodes(inventory))
        if workflow is None:
            session = WorkflowSession(catalog, inventory, models=models)
        else:
            session = WorkflowSession.from_api(workflow, catalog, inventory, models=models)
        # A generation pipeline is offered only when a model on the server can run it.
        capabilities = {capability for model in models for capability in model.capabilities}
        chosen = tuple(
            pipeline
            for pipeline in (available_pipelines(models) if pipelines is None else pipelines)
            if pipeline.CAPABILITY is None or pipeline.CAPABILITY in capabilities
        )
        if chat_model is None:
            key = api_key or load_settings(["ANTHROPIC_API_KEY"], env_file)["ANTHROPIC_API_KEY"]
            chat_model = build_chat_model(
                model, api_key=key, effort=effort, refusal_fallback=refusal_fallback
            )
        profile = getattr(chat_model, "profile", None) or {}
        # The turn's context goes in a system message when the model takes one mid
        # conversation, so it never mixes with the user's words.
        self.__system_context = bool(profile.get("mid_conversation_system_messages"))
        self.__session = session
        self.__vision = vision
        self.__turn = asyncio.Lock()
        self.__turns = 0
        self.__graph = create_agent(
            chat_model,
            build_tools(session, chosen),
            system_prompt=system_prompt(catalog, inventory, chosen, models),
            middleware=[
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                ValidationGate(session),
                ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
            ],
            checkpointer=InMemorySaver(),
        )
        self.__config = {"configurable": {"thread_id": "workflow"}, "recursion_limit": 10 * max_model_calls}

    @classmethod
    def from_workflow(
        cls, workflow: Mapping[str, Any], inventory: ServerInventory, **options: Any
    ) -> "WorkflowAgent":
        """An agent that edits a workflow exported with ComfyUI's Export (API)."""
        return cls(inventory, workflow=workflow, **options)

    @property
    def workflow(self) -> dict[str, Any]:
        return self.__session.snapshot()

    @property
    def previous_workflows(self) -> list[dict[str, Any]]:
        """The workflows the agent set aside to start a new one, oldest first."""
        return copy.deepcopy(self.__session.previous)

    @property
    def files(self) -> dict[str, bytes]:
        """The attached files the workflow loads, to send to the server with it."""
        session = self.__session
        names = [name for name in _referenced_files(self.workflow) if name in session.files]
        return session.files.payload(names)

    def view(self) -> str:
        return self.__session.view()

    def validate(self) -> ValidationReport:
        return self.__session.validate()

    async def send(self, text: str, files: Mapping[str, bytes] | None = None) -> AgentReply:
        async with self.__turn:
            session = self.__session
            attached = [session.files.add(name, data) for name, data in (files or {}).items()]
            before = session.snapshot()
            session.start_turn()
            self.__turns += 1
            turn_id = f"user-turn-{self.__turns}"
            try:
                result = await self.__graph.ainvoke(
                    {"messages": self.__messages(text, attached, turn_id)}, self.__config
                )
            except (anthropic.APIError, GraphRecursionError) as error:
                logger.warning("The agent turn failed: %s", error)
                return self.__reply("error", f"The agent could not finish the turn: {error}", before)
            messages = result["messages"]
            start = next((index for index, message in enumerate(messages) if message.id == turn_id), 0)
            usage = _usage(messages[start:])
            last = messages[-1]
            if session.question is not None:
                return self.__reply("needs_input", session.question.text, before, usage)
            if isinstance(last, AIMessage) and last.response_metadata.get("stop_reason") == "refusal":
                return self.__reply("refused", "The model declined this request.", before, usage)
            return self.__reply(None, _final_text(messages), before, usage)

    def __reply(
        self,
        status: Status | None,
        message: str,
        before: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> AgentReply:
        session = self.__session
        workflow = session.snapshot()
        changes = diff_workflows(before, workflow)
        report = session.validate()
        if status is None:
            status = "invalid" if changes and not report.ok else "completed"
        return AgentReply(
            status=status,
            message=message or changes.text(),
            workflow=workflow,
            question=session.question,
            changes=changes,
            validation=report,
            input_files=_referenced_files(workflow),
            usage=usage or {},
        )

    def __messages(self, text: str, attached: list[InputFile], turn_id: str) -> list[BaseMessage]:
        images = [file for file in attached if file.kind == "image"] if self.__vision else []
        blocks: list[dict[str, Any]] = []
        for file in images:
            blocks.append({"type": "text", "text": f"Attached image {file.name}:"})
            blocks.append(vision_block(file))
        context = self.__context(attached)
        if self.__system_context:
            content: Any = [*blocks, {"type": "text", "text": text}] if blocks else text
            return [HumanMessage(content, id=turn_id), SystemMessage(context)]
        content = [{"type": "text", "text": context}, *blocks, {"type": "text", "text": text}]
        return [HumanMessage(content, id=turn_id)]

    def __context(self, attached: list[InputFile]) -> str:
        session = self.__session
        now = ", ".join(file.describe() for file in attached) or "none"
        lines = [
            "Application context for this turn (not written by the user):",
            f"- Files attached with this message: {now}.",
        ]
        earlier = [file.describe() for file in session.files if file not in attached]
        if earlier:
            lines.append(f"- Files attached earlier: {', '.join(earlier)}.")
        count = len(session.workflow.nodes)
        if count:
            outputs = ", ".join(session.output_node_ids()) or "none"
            lines.append(
                f"- Current workflow: {count} nodes; output nodes: {outputs}. "
                "Call view_workflow before changing it."
            )
        else:
            lines.append("- Current workflow: empty.")
        return "\n".join(lines)
