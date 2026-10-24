"""Without RunPod: lists the model files of your ComfyUI by hand and builds for them.

    python examples/describe_your_server.py
    python examples/describe_your_server.py http://127.0.0.1:8188

Folders left out are not checked; the files of a listed folder are, and an empty folder
means the server has none. With the address of a ComfyUI, the workflow is queued there.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

from agent_comfy_ui_api_workflow import ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parent / "output"
# The files in ComfyUI/models/<folder> on your server.
INVENTORY = ServerInventory(
    {
        "checkpoints": ["sd_xl_base_1.0.safetensors"],
        "upscale_models": ["RealESRGAN_x4plus.safetensors"],
        "loras": [],
    }
)


async def main(server: str | None) -> None:
    agent = WorkflowAgent(INVENTORY)

    reply = await agent.send("a cozy cabin in a snowy forest at night, then make it 2 times bigger")

    print(f"[{reply.status}] {reply.message}")
    if reply.status != "completed":
        return
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "cabin_workflow.json"
    path.write_text(json.dumps(reply.workflow, indent=2), encoding="utf-8")
    print(f"Saved in {path}")
    if server is not None:
        async with httpx.AsyncClient(base_url=server) as comfy:
            response = await comfy.post("/prompt", json={"prompt": reply.workflow})
        response.raise_for_status()
        print(f"Queued on {server}: prompt {response.json()['prompt_id']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
