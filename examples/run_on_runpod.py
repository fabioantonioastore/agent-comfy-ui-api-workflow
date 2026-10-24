"""Builds a workflow and runs it on the RunPod endpoint in .env, saving what it makes.

    python examples/run_on_runpod.py
    python examples/run_on_runpod.py "a 10 second acoustic guitar song"
    python examples/run_on_runpod.py "cut this video from second 2 to 5" trip.mp4
    python examples/run_on_runpod.py "make this photo sing this song" me.jpg song.mp3

The files after the request are attached. It costs Claude tokens and GPU time.
"""

import asyncio
import sys
from pathlib import Path

from agent_comfy_ui_api_workflow import RunPodClient, ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parent / "output"


async def main(request: str, paths: list[Path]) -> None:
    agent = WorkflowAgent(ServerInventory.load(OUTPUT / "inventory.json"))
    reply = await agent.send(request, files={path.name: path.read_bytes() for path in paths})
    print(f"[{reply.status}] {reply.message}")
    if reply.status != "completed":
        return

    async with RunPodClient.from_env() as runpod:
        # Video models can take several minutes.
        job = await runpod.run(reply.workflow, files=agent.files, timeout=1800)
    if not job.ok:
        sys.exit(f"The job failed: {job.error}")
    for file in job.files:
        if file.filename not in agent.files:  # the worker also returns some of the inputs
            (OUTPUT / file.filename).write_bytes(file.data)
            print(f"Saved {OUTPUT / file.filename}")
    print(f"GPU time: {(job.execution_ms or 0) / 1000:.0f} s")


if __name__ == "__main__":
    request = sys.argv[1] if len(sys.argv) > 1 else "a red apple on a wooden table, studio photo"
    asyncio.run(main(request, [Path(arg) for arg in sys.argv[2:]]))
