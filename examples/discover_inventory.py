"""Asks the RunPod endpoint in .env which model files its ComfyUI has, and saves the list.

The other examples load examples/output/inventory.json: run this one first, and again
when the server's models change. It costs one short job on the endpoint.

    python examples/discover_inventory.py
"""

import asyncio
from pathlib import Path

from agent_comfy_ui_api_workflow import RunPodClient

INVENTORY = Path(__file__).resolve().parent / "output" / "inventory.json"


async def main() -> None:
    async with RunPodClient.from_env() as runpod:
        inventory = await runpod.discover_inventory()
    INVENTORY.parent.mkdir(exist_ok=True)
    inventory.save(INVENTORY)
    print(inventory.describe())
    print(f"\nSaved in {INVENTORY}")


if __name__ == "__main__":
    asyncio.run(main())
