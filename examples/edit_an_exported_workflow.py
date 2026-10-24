"""Changes a workflow exported from ComfyUI with Export (API), keeping its node ids.

    python examples/edit_an_exported_workflow.py
    python examples/edit_an_exported_workflow.py workflow_api.json "use 30 steps"

With no arguments it edits examples/output/workflow_api.json, the workflow that
build_a_workflow.py, or /save in chat.py, saved last.
"""

import asyncio
import json
import sys
from pathlib import Path

from agent_comfy_ui_api_workflow import ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parent / "output"


async def main(path: Path, request: str) -> None:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    agent = WorkflowAgent.from_workflow(workflow, ServerInventory.load(OUTPUT / "inventory.json"))

    reply = await agent.send(request)

    print(f"[{reply.status}] {reply.message}")
    print(reply.changes.text())
    if reply.status == "completed":
        edited = OUTPUT / f"{path.stem}_edited.json"
        edited.write_text(json.dumps(reply.workflow, indent=2), encoding="utf-8")
        print(f"Saved in {edited}")


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT / "workflow_api.json"
    asyncio.run(main(source, " ".join(sys.argv[2:]) or "make it portrait 9:16 and use 30 steps"))
