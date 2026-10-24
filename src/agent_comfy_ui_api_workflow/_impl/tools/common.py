import logging
from collections.abc import Awaitable, Callable
from typing import Any

from comfy_ui_api_workflow.exceptions import ComfyUIAPIWorkflowException
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ValidationError

from ..errors import AgentComfyUIError

logger = logging.getLogger(__name__)

LINK_SYNTAX = 'a link to an output, written "<node_id>.<output name>" like "VAEDecode_5.IMAGE"'


def _invalid_arguments(error: ValidationError) -> str:
    # LangChain's own message says only that the input was invalid.
    problems = "; ".join(
        f"{'.'.join(map(str, item['loc'])) or 'arguments'}: {item['msg']}" for item in error.errors()
    )
    return f"Invalid arguments: {problems}. Fix them and call the tool again."


def make_tool(
    function: Callable[..., Awaitable[str]],
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    return_direct: bool = False,
) -> StructuredTool:
    """A tool whose expected errors go back to the model as tool errors it can fix."""

    async def run(**arguments: Any) -> str:
        try:
            return await function(**arguments)
        except (ComfyUIAPIWorkflowException, AgentComfyUIError) as error:
            raise ToolException(str(error)) from error
        except Exception as error:
            # A bug in a tool must still answer the call, or the conversation is left
            # with a tool use that has no result. The traceback goes to the log.
            logger.exception("The tool %s failed", name)
            raise ToolException(
                f"The tool failed unexpectedly ({type(error).__name__}: {error}). "
                "Try another way to do this step."
            ) from error

    return StructuredTool.from_function(
        coroutine=run,
        name=name,
        description=description,
        args_schema=args_schema,
        handle_tool_error=True,
        handle_validation_error=_invalid_arguments,
        return_direct=return_direct,
    )
