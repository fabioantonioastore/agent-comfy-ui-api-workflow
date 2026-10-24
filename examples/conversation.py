"""Several turns with one agent: each request changes what the agent built so far.

    python examples/conversation.py
"""

import asyncio
from pathlib import Path

from agent_comfy_ui_api_workflow import ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parent / "output"
TURNS = (
    "an astronaut cat floating in space, digital art",
    "make it 16:9 and use 30 steps",
    "now a new one: a 3 second video of waves on a beach",
)


async def main() -> None:
    agent = WorkflowAgent(ServerInventory.load(OUTPUT / "inventory.json"))
    for request in TURNS:
        reply = await agent.send(request)
        print(f"> {request}\n[{reply.status}] {reply.message}\n{reply.changes.text()}\n")
    # Asking for a new one sets the current workflow aside instead of changing it.
    print(f"Set aside: {len(agent.previous_workflows)}; the current one has {len(agent.workflow)} nodes")


if __name__ == "__main__":
    asyncio.run(main())
