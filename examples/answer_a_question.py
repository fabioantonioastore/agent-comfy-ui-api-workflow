"""A request that needs a file the user did not attach comes back as a question.

    python examples/answer_a_question.py me.jpg

The first turn asks for the photo and the second sends it. The workflow loads the photo
by the name in reply.input_files: upload it with the workflow.
"""

import asyncio
import json
import sys
from pathlib import Path

from agent_comfy_ui_api_workflow import ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parent / "output"


async def main(photo: Path) -> None:
    agent = WorkflowAgent(ServerInventory.load(OUTPUT / "inventory.json"))

    reply = await agent.send("make an elephant with my face")
    print(f"[{reply.status}] {reply.message}")
    if reply.status == "needs_input":
        print(f"It needs: {', '.join(reply.question.needs)}\n")
        reply = await agent.send("here it is", files={photo.name: photo.read_bytes()})
        print(f"[{reply.status}] {reply.message}")

    if reply.status == "completed":
        path = OUTPUT / "elephant_workflow.json"
        path.write_text(json.dumps(reply.workflow, indent=2), encoding="utf-8")
        print(f"\nSaved in {path}; upload with it: {', '.join(reply.input_files)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/answer_a_question.py PHOTO")
    asyncio.run(main(Path(sys.argv[1])))
