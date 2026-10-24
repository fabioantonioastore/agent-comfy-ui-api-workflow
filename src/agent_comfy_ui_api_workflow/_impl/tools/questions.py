from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..session import Question, WorkflowSession
from .common import make_tool

Need = Literal["image", "video", "audio", "mask", "answer"]


class AskUserArguments(BaseModel):
    question: str = Field(
        description="The question, in the user's language: what you need and why."
    )
    needs: list[Need] = Field(
        default_factory=lambda: ["answer"],
        description=(
            "What the user should send back: files to attach (image, video, audio, "
            "mask) and/or an answer in text."
        ),
    )


def ask_user_tool(session: WorkflowSession) -> StructuredTool:
    async def ask_user(question: str, needs: list[Need] | None = None) -> str:
        session.question = Question(question.strip(), tuple(needs or ("answer",)))
        return "The question was sent to the user. Stop now: end your turn without calling any other tool."

    return make_tool(
        ask_user,
        name="ask_user",
        description=(
            "Asks the user for what you need and ends your turn. Use it when the request "
            "needs a file the user did not attach (an image, video, audio or mask), when "
            "it asks for something the server's models cannot do, or when it is ambiguous "
            "in a way that changes the result. Call it alone, as the only tool call in "
            "your message."
        ),
        args_schema=AskUserArguments,
        return_direct=True,
    )
