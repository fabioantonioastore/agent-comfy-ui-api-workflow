from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from .session import WorkflowSession

NOTICE = "[Automatic check by the application, not a message from the user]"


class ValidationGate(AgentMiddleware):
    """Sends the model back to fix the workflow when it finishes a turn with problems."""

    def __init__(self, session: WorkflowSession, max_nudges: int = 2) -> None:
        super().__init__()
        self.session = session
        self.max_nudges = max_nudges

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        session = self.session
        if (
            last.response_metadata.get("stop_reason") == "refusal"
            or session.question is not None
            or not session.changed
            or session.nudges >= self.max_nudges
        ):
            return None
        report = session.validate()
        if report.ok:
            return None
        session.nudges += 1
        # After the model's own answer only a user turn may follow, so the notice is one.
        notice = f"{NOTICE}\n{report.text()}\nFix these problems with your tools, then finish again."
        return {"messages": [HumanMessage(notice)], "jump_to": "model"}
