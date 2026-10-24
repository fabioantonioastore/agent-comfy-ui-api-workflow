"""Builds a workflow from one request and saves it in the API format.

    python examples/build_a_workflow.py
    python examples/build_a_workflow.py "a 5 second video of a lion running, portrait"

It calls Claude, not the GPU: the workflow is only built. To run it, see
run_on_runpod.py, or send it to the POST /prompt of your ComfyUI.
"""

import asyncio
import json
import sys
from pathlib import Path

from agent_comfy_ui_api_workflow import ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parent / "output"


async def main(request: str) -> None:
    agent = WorkflowAgent(ServerInventory.load(OUTPUT / "inventory.json"))

    reply = await agent.send(request)

    print(f"[{reply.status}] {reply.message}\n")
    if reply.status == "invalid":
        print(reply.validation.text())
    if reply.status != "completed":
        return
    print(agent.view())
    path = OUTPUT / "workflow_api.json"
    path.write_text(json.dumps(reply.workflow, indent=2), encoding="utf-8")
    print(f"\nSaved in {path}")
    print(f"Tokens: {reply.usage}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "a watercolor of a lighthouse at dusk, widescreen"))
