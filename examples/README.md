# Examples

Run them from the root of the repository, where your `.env` is: `ANTHROPIC_API_KEY` for
the agent, and `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` for RunPod. What they save goes
to `examples/output/`.

| Example | What it shows | What it costs |
|---|---|---|
| [`discover_inventory.py`](discover_inventory.py) | Reads the model files of your RunPod endpoint into `output/inventory.json`, which the others load. Run it first. | One short job |
| [`build_a_workflow.py`](build_a_workflow.py) | A request becomes a workflow in the API format | Claude tokens |
| [`answer_a_question.py`](answer_a_question.py) | A request that needs a photo comes back as a question; the next turn attaches it | Claude tokens |
| [`edit_an_exported_workflow.py`](edit_an_exported_workflow.py) | Changes a workflow exported from ComfyUI and keeps its node ids | Claude tokens |
| [`conversation.py`](conversation.py) | Turns that change the same workflow, then a new one | Claude tokens |
| [`run_on_runpod.py`](run_on_runpod.py) | Builds any request, with attached files, runs it on RunPod and saves the results | Claude tokens and GPU time |
| [`chat.py`](chat.py) | Talks to the agent in the terminal: attach files with `@path`, run with `/run` | Claude tokens and GPU time |
| [`describe_your_server.py`](describe_your_server.py) | Without RunPod: the server's files listed by hand, and the workflow queued on your ComfyUI | Claude tokens |

Some requests to try, as far as the models of your server go:

```bash
python examples/run_on_runpod.py "a portrait photo of an old fisherman, 3:4"
python examples/run_on_runpod.py "a 5 second video of a paper boat on a calm lake"
python examples/run_on_runpod.py "rain on a tin roof, with the sound of the rain, 5 seconds"
python examples/run_on_runpod.py "a 30 second lo-fi hip hop beat"
python examples/run_on_runpod.py "make this photo sing this song" me.jpg song.mp3
python examples/run_on_runpod.py "cut this video from second 2 to 5" trip.mp4
python examples/run_on_runpod.py "make this image 4 times bigger" logo.png
```

Requests can be in any language; the agent answers in the language of the request.
