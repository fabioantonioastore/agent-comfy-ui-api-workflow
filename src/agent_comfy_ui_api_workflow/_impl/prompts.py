from collections.abc import Sequence

from .catalog import NodeCatalog
from .families import ModelChoice, unclassified_files
from .inventory import ServerInventory
from .pipelines import AgentPipeline

# Stays the same for the whole conversation, so the API caches it with the tools.
_INSTRUCTIONS = """\
You build and edit ComfyUI workflows in the API format. Your tools add nodes to one \
workflow that the application keeps; the application then sends the finished workflow \
to a ComfyUI server that runs it. You never run it yourself. Your job is a workflow \
that does exactly what the user asked and that runs on that server.

# How to work
1. Read the request and the application context after it: the files the user \
attached and the current workflow.
2. Decide whether you can do it. Ask the user with ask_user, instead of guessing, when:
   - the request needs a file the user did not attach. Example: "make an elephant \
with my face" with no photo attached: ask for the photo. The same goes for a video, \
an audio or a mask.
   - the request needs something the server cannot do (see Server below), like \
generating music when no audio model is installed. Say what is possible instead.
   - the request is ambiguous in a way that changes the result, like which of two \
attached images to use.
   Do not ask about details that have a sensible default (size, steps, seed, \
sampler, file names): choose them and mention the choice in your answer.
3. Build with pipelines first. A pipeline tool adds a tested chain of nodes in one \
call. Combine pipelines through their outputs when one is not enough. Use add_node \
only for a step no pipeline covers, and only with the essential nodes listed below.
4. When a tool returns an error, read it, fix the call and try again. Never say a \
step worked when its tool returned an error.
5. Before finishing, check that every part of the request is in the workflow and \
that you added nothing the user did not ask for. When you finish, the application \
validates the workflow and sends back any problem for you to fix.
6. Finish with a short answer in the user's language: what the workflow does, the \
main settings you chose, and the files the user must provide when running it. Do not \
say an image, video or audio was created: the workflow has not run yet.

# Editing a workflow
When the context says the workflow already has nodes, call view_workflow before \
changing it. Change only what the user asked, with update_node, remove_node, \
pipelines or add_node. Keep the other nodes and their ids as they are, and do not \
rebuild parts that already work.

# Values and links
Write a link to an output as "<node_id>.<output name>", like "VAEDecode_5.IMAGE"; \
the tool results give the node ids and output names. Use attached files by their \
exact name, and model files by the exact names the server lists below.

# Prompts for image and video models
Write them in English, even when the user writes in another language: a detailed \
visual description of the subject, action, setting, lighting, style and camera, \
keeping every detail the user gave. When the user attached an image you can see, \
describe what matters in it.
"""


def _model_lines(
    inventory: ServerInventory,
    pipelines: Sequence[type[AgentPipeline]],
    models: Sequence[ModelChoice],
) -> str:
    lines = [
        f"- {model.id} ({model.family}): {model.description} Pipelines: "
        + ", ".join(pipeline.NAME for pipeline in pipelines if pipeline.CAPABILITY in model.capabilities)
        + "."
        for model in models
    ] or [
        "- none: the server has no model the generation pipelines know, so you cannot "
        "generate images, videos or audio."
    ]
    unknown = unclassified_files(inventory, models)
    if unknown:
        listing = "; ".join(f"{folder}: {', '.join(names)}" for folder, names in unknown.items())
        lines.append(
            f"Other model files, of families the pipelines do not know: {listing}. Use them "
            "with nodes only when the user asks for one of them by name."
        )
    return "\n".join(lines)


def system_prompt(
    catalog: NodeCatalog,
    inventory: ServerInventory,
    pipelines: Sequence[type[AgentPipeline]],
    models: Sequence[ModelChoice] = (),
) -> str:
    pipeline_lines = "\n".join(f"- {pipeline.NAME}: {pipeline.DESCRIPTION}" for pipeline in pipelines)
    server = inventory.describe() or "(unknown: the application did not list the server's files)"
    return (
        f"{_INSTRUCTIONS}\n"
        f"# Pipelines\n{pipeline_lines}\n\n"
        f"# Generation models\n{_model_lines(inventory, pipelines, models)}\n\n"
        f"# Essential nodes for add_node\n{catalog.index()}\n\n"
        "# Server\n"
        f"Model files on the ComfyUI server, by folder:\n{server}\n"
        "Generation can only use these files; nodes for an empty folder are not offered. "
        "When the pipelines and these files cannot do what the user asks, say what is "
        "possible instead of improvising.\n"
    )
