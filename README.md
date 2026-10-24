# agent-comfy-ui-api-workflow

An async agent on Claude (through LangChain) that builds, edits and validates
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows in the API format, with
[comfy-ui-api-workflow](https://github.com/fabioantonioastore/comfy-ui-api-workflow).

- **Pipelines first.** The agent adds tested chains of nodes in one call:
  `text_to_image`, `image_to_image`, `inpaint`, `outpaint`, `text_to_video`,
  `image_to_video` (also between a start and an end image), `sound_to_video` (a photo
  that talks or sings an audio), `text_to_audio` (music), `upscale_image_with_model`,
  and video, audio and image editing. Single nodes come from a short list of essential
  ones, only for steps no pipeline covers.
- **Your server's models only.** Generation pipelines are offered for the model
  families found on the server, and every model name is checked against the server's
  files. The families it knows:

  | Family | Pipelines |
  |---|---|
  | SDXL | images: text to image, image to image, inpaint, outpaint |
  | Wan 2.2 TI2V 5B | video from text or an image, 24 fps |
  | Wan 2.2 A14B T2V / I2V, with the LightX2V 4-step LoRAs | video from text, an image, or a start and an end image, 16 fps |
  | MiniMax H3 (with its turbo LoRA) | video with its own sound: speech, effects, music; 5 to 15 s |
  | Wan 2.2 S2V | a photo animated in sync with an audio |
  | ACE-Step 1.5 XL turbo | music, sung or instrumental |
  | Upscale models (Real-ESRGAN...) | image upscaling |

  MiniMax H3 has its own community license, which restricts where it may be used:
  read it before serving that model.
- **It asks when it must.** A request that needs a file the user did not attach (an
  image, a video, an audio, a mask), something the server cannot do, or a choice that
  changes the result, comes back as a question instead of a guess.
- **Checked before it is delivered.** When the agent finishes, the workflow is
  validated: an output node, every node connected, model files the server has, files
  the user attached, and a JSON that loads back. Problems go back to the agent to fix.
- **It never runs the workflow.** You get the JSON for `POST /prompt` and the list of
  files to upload with it.

## Installation

```bash
pip install git+https://github.com/fabioantonioastore/agent-comfy-ui-api-workflow
```

The agent reads `ANTHROPIC_API_KEY`, and `RunPodClient` reads `RUNPOD_API_KEY` and
`RUNPOD_ENDPOINT_ID`, from the environment or from a `.env` file (`NAME=value` or
`NAME: value` lines).

## Quick start

```python
import asyncio
from pathlib import Path

from agent_comfy_ui_api_workflow import RunPodClient, ServerInventory, WorkflowAgent


async def main():
    # Once: ask the server which model files it has, and keep the answer. Workers that
    # answer {"object_info": true} jobs give every loader's files; the official worker
    # gets a probe instead, which misses the loaders its handler does not check.
    async with RunPodClient.from_env() as runpod:
        inventory = await runpod.discover_inventory()
    inventory.save("inventory.json")  # later: ServerInventory.load("inventory.json")

    agent = WorkflowAgent(inventory)
    reply = await agent.send("make an elephant with my face")
    if reply.status == "needs_input":
        print(reply.question.text, reply.question.needs)  # ... ('image',)
        photo = Path("me.jpg").read_bytes()
        reply = await agent.send("here it is", files={"me.jpg": photo})

    print(reply.status, reply.message)
    print(reply.workflow)     # the API format, ready for POST /prompt
    print(reply.input_files)  # ('me.jpg',): upload it with the workflow


asyncio.run(main())
```

Each `send` is one turn of the same conversation, so the next request can change what
the agent built: "now make it 16:9 with 35 steps".

More in [examples/](examples/): questions, editing, conversations, running on RunPod,
a chat in the terminal and a server described by hand.

## Editing an exported workflow

```python
import json

workflow = json.loads(Path("workflow_api.json").read_text())  # ComfyUI's Export (API)
agent = WorkflowAgent.from_workflow(workflow, inventory)
reply = await agent.send("change the prompt to a lighthouse at dusk and use 30 steps")
print(reply.changes.text())  # Changed: 6 (text), 3 (steps).
```

The agent reads the workflow before changing it, changes only what was asked and keeps
the node ids.

## Replies

`send` returns an `AgentReply`:

| Field | What it holds |
|---|---|
| `status` | `completed`, `needs_input` (a question), `invalid` (problems the agent could not fix), `refused` or `error` |
| `message` | The agent's answer, in the user's language |
| `workflow` | The workflow in the API format, as plain JSON |
| `question` | For `needs_input`: `text` and `needs` (`image`, `video`, `audio`, `mask`, `answer`) |
| `changes` | Nodes added, removed and changed in this turn |
| `validation` | The checks above: `ok`, `problems`, `text()` |
| `input_files` | The user files the workflow loads |
| `usage` | Tokens of the turn: input, output, cache reads and writes, model calls |

`agent.workflow`, `agent.files` (the attached files the workflow loads),
`agent.previous_workflows`, `agent.view()` and `agent.validate()` give the same
information between turns.

## Running a workflow

The agent only builds workflows. To run one on a RunPod `worker-comfyui` endpoint:

```python
async with RunPodClient.from_env() as runpod:
    job = await runpod.run(reply.workflow, files=agent.files)
if not job.ok:
    raise RuntimeError(job.error)
for file in job.files:
    if file.filename not in agent.files:  # the worker also returns some of the inputs
        Path(file.filename).write_bytes(file.data)
```

`job.ok` is false when a node failed, even when RunPod reports the job `COMPLETED`;
`job.error` then names the node and the reason. RunPod takes requests of up to 10 MiB,
and the files go in them base64-encoded: about 7.5 MB of files per job.

`run` waits up to `timeout` seconds (600 by default) and then cancels the job. When
RunPod's API fails for a moment while the job runs, it asks for the status again.

## Configuration

```python
WorkflowAgent(
    inventory,
    model="claude-opus-5-5",  # any Claude model id
    effort="high",            # low, medium, high, xhigh or max
    api_key=None,             # default: ANTHROPIC_API_KEY from the environment or env_file
    env_file=".env",
    vision=True,              # show attached images to the model
    refusal_fallback=True,    # let the API retry a declined request on another model
    max_model_calls=40,       # per turn
)
```

## Development

```bash
pip install -e ".[dev]"
pytest              # offline tests
pytest --anthropic  # the agent on the Claude API (paid tokens)
pytest --runpod     # every pipeline on the RunPod endpoint (paid GPU time)
pytest --runpod --anthropic  # everything, plus the agent's workflows run on the endpoint
pytest --cov=agent_comfy_ui_api_workflow --cov-report=term-missing  # coverage
```

## License

MIT
