import asyncio
import json
from pathlib import Path

from agent_comfy_ui_api_workflow import ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parents[1] / "examples" / "output"
TURNS = (
    "Create a video of an astronaut with my face walking in the moon",
    "Now make the astronaut an elephant astronaut",
)


async def main() -> None:
    agent = WorkflowAgent(ServerInventory.load(OUTPUT / "inventory.json"))
    await agent.send("Create a video of an astronaut walking in the moon")
    print(agent.view())
    path = OUTPUT / "astronaut_elephant.json"
    path.write_text(json.dumps(agent.workflow, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
