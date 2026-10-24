"""Talks to the agent in the terminal, attaches files and runs the result on RunPod.

    python examples/chat.py
    > make this photo sing this song @me.jpg @song.mp3
    > make it portrait
    > /run

A word that starts with @ attaches that file. /run runs the workflow on the RunPod
endpoint in .env, /save saves it, /view shows it and /quit ends.
"""

import asyncio
import json
import re
from pathlib import Path

from agent_comfy_ui_api_workflow import AgentComfyUIError, RunPodClient, ServerInventory, WorkflowAgent

OUTPUT = Path(__file__).resolve().parent / "output"
ATTACHMENT = re.compile(r"(?<!\S)@(\S+)")


async def run(agent: WorkflowAgent) -> None:
    report = agent.validate()
    if not report.ok:
        print(report.text())
        return
    async with RunPodClient.from_env() as runpod:
        job = await runpod.run(agent.workflow, files=agent.files, timeout=1800)
    if not job.ok:
        print(f"The job failed: {job.error}")
        return
    for file in job.files:
        if file.filename not in agent.files:  # the worker also returns some of the inputs
            (OUTPUT / file.filename).write_bytes(file.data)
            print(f"Saved {OUTPUT / file.filename}")


async def send(agent: WorkflowAgent, line: str) -> None:
    try:
        files = {Path(name).name: Path(name).expanduser().read_bytes() for name in ATTACHMENT.findall(line)}
        reply = await agent.send(ATTACHMENT.sub("", line).strip(), files=files)
    except (OSError, AgentComfyUIError) as error:
        print(error)
        return
    print(f"[{reply.status}] {reply.message}")


async def main() -> None:
    agent = WorkflowAgent(ServerInventory.load(OUTPUT / "inventory.json"))
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line == "/quit":
            break
        if line == "/run":
            await run(agent)
        elif line == "/save":
            path = OUTPUT / "workflow_api.json"
            path.write_text(json.dumps(agent.workflow, indent=2), encoding="utf-8")
            print(f"Saved in {path}")
        elif line == "/view":
            print(agent.view())
        elif line:
            await send(agent, line)


if __name__ == "__main__":
    asyncio.run(main())
