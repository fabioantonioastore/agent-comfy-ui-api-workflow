import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_comfy_ui_api_workflow import WorkflowAgent
from agent_comfy_ui_api_workflow._impl.errors import SettingsError
from agent_comfy_ui_api_workflow._impl.inventory import ServerInventory
from conftest import SDXL
from media import MP4, png
from scripted import ScriptedChatModel, answer, calls

PROFILE = {"mid_conversation_system_messages": True}


def agent_with(inventory, *replies, **options) -> tuple[WorkflowAgent, ScriptedChatModel]:
    model = ScriptedChatModel(replies=list(replies), profile=PROFILE)
    return WorkflowAgent(inventory, chat_model=model, **options), model


async def test_builds_a_workflow_with_a_pipeline(inventory):
    agent, model = agent_with(
        inventory,
        calls(("trim_video", {"video": "clip.mp4", "start_seconds": 1.0, "duration_seconds": 2.0})),
        answer("Pronto: o workflow corta o vídeo de 1 s a 3 s."),
    )

    reply = await agent.send("corte meu vídeo do segundo 1 ao 3", files={"clip.mp4": MP4})

    assert reply.status == "completed"
    assert reply.message == "Pronto: o workflow corta o vídeo de 1 s a 3 s."
    assert {node["class_type"] for node in reply.workflow.values()} == {
        "LoadVideo",
        "Video Slice",
        "SaveVideo",
    }
    assert reply.input_files == ("clip.mp4",)
    assert agent.files == {"clip.mp4": MP4}
    assert reply.changes.added == ("LoadVideo_0", "VideoSlice_1", "SaveVideo_2")
    assert reply.validation.ok


async def test_asks_for_a_missing_file_and_goes_on_with_the_answer(inventory):
    agent, model = agent_with(
        inventory,
        calls(("ask_user", {"question": "Envie uma foto do seu rosto.", "needs": ["image"]})),
        calls(("upscale_image", {"image": "rosto.png", "scale": 2.0})),
        answer("Feito."),
    )

    first = await agent.send("crie um elefante com a minha cara")

    assert first.status == "needs_input"
    assert first.message == "Envie uma foto do seu rosto."
    assert first.question.needs == ("image",)
    assert len(model.received) == 1

    second = await agent.send("aqui está", files={"rosto.png": png()})

    assert second.status == "completed"
    sent = model.received[1]
    assert isinstance(sent[-3], ToolMessage)
    assert isinstance(sent[-2], HumanMessage)
    assert isinstance(sent[-1], SystemMessage)
    assert "rosto.png (image, 64x48)" in sent[-1].content


async def test_the_validation_gate_sends_the_model_back_to_fix_the_workflow(inventory):
    agent, model = agent_with(
        inventory,
        calls(("add_node", {"node_type": "LoadVideo", "inputs": {"file": "clip.mp4"}})),
        answer("Feito."),
        calls(("save_output", {"source": "LoadVideo_0.VIDEO"})),
        answer("Agora sim."),
    )

    reply = await agent.send("salve meu vídeo", files={"clip.mp4": MP4})

    assert reply.status == "completed"
    assert reply.message == "Agora sim."
    notice = model.received[2][-1]
    assert isinstance(notice, HumanMessage)
    assert "not a message from the user" in notice.content
    assert "no output node" in notice.content


async def test_the_gate_gives_up_after_two_tries(inventory):
    agent, model = agent_with(
        inventory,
        calls(("add_node", {"node_type": "EmptyImage", "inputs": {"width": 64, "height": 64}})),
        answer("Feito."),
        answer("Feito de novo."),
        answer("Não consigo."),
    )

    reply = await agent.send("crie uma imagem vazia")

    assert reply.status == "invalid"
    assert not reply.validation.ok
    assert len(model.received) == 4


async def test_a_turn_that_changes_nothing_is_completed(inventory):
    agent, _ = agent_with(inventory, answer("Este workflow está vazio."))

    reply = await agent.send("o que este workflow faz?")

    assert reply.status == "completed"
    assert not reply.changes


async def test_edits_a_loaded_workflow_and_reports_only_what_changed(inventory):
    loaded = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
        "9": {"class_type": "PreviewAny", "inputs": {"source": ["6", 0]}},
    }
    agent, model = agent_with(
        inventory,
        calls(("update_node", {"node_id": "6", "inputs": {"text": "a dog"}})),
        answer("Troquei o prompt."),
        workflow=loaded,
    )

    reply = await agent.send("troque o gato por um cachorro")

    assert reply.changes.changed == {"6": ("text",)}
    assert reply.workflow["6"]["inputs"]["text"] == "a dog"
    assert reply.workflow["4"] == loaded["4"]
    context = model.received[0][-1]
    assert "Current workflow: 3 nodes" in context.content
    assert "view_workflow" in context.content


async def test_attached_images_are_shown_to_the_model(inventory):
    agent, model = agent_with(inventory, answer("Vi a foto."))

    await agent.send("o que tem na foto?", files={"foto.png": png()})

    content = model.received[0][-2].content
    assert content[0] == {"type": "text", "text": "Attached image foto.png:"}
    assert content[1]["type"] == "image"
    assert content[-1] == {"type": "text", "text": "o que tem na foto?"}


async def test_the_system_prompt_has_the_pipelines_nodes_and_server(inventory):
    agent, model = agent_with(inventory, answer("Ok."))

    await agent.send("oi")

    system = model.received[0][0]
    assert isinstance(system, SystemMessage)
    assert "- trim_video:" in system.content
    assert "- KSampler(" in system.content
    assert "- Wan22ImageToVideoLatent(" in system.content
    assert "- sd_xl_base_1.0 (sdxl):" in system.content
    assert "Pipelines: text_to_video, image_to_video." in system.content
    assert SDXL in system.content
    assert model.tool_names[:2] == ["text_to_image", "image_to_image"]
    assert "ask_user" in model.tool_names


async def test_unknown_model_files_are_listed_for_use_by_name(inventory):
    with_unknown = inventory.to_dict()
    with_unknown["checkpoints"].append("mystery_model.safetensors")
    agent, model = agent_with(type(inventory).from_dict(with_unknown), answer("Ok."))

    await agent.send("oi")

    assert "mystery_model.safetensors" in model.received[0][0].content.split("# Generation models")[1]


async def test_nodes_for_empty_server_folders_are_hidden(inventory):
    agent, model = agent_with(inventory, answer("Ok."))

    await agent.send("oi")

    assert "- LoraLoader(" not in model.received[0][0].content
    assert "- CheckpointLoaderSimple(" in model.received[0][0].content


async def test_the_reply_adds_up_the_tokens_of_the_turn(inventory):
    usage = {
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
        "input_token_details": {"cache_read": 80, "cache_creation": 20},
    }
    first = calls(("view_workflow", {}))
    first.usage_metadata = usage
    second = answer("Vazio.")
    second.usage_metadata = usage
    agent, _ = agent_with(inventory, answer("Oi."), first, second)
    await agent.send("oi")

    reply = await agent.send("o que tem no workflow?")

    assert reply.usage == {
        "input_tokens": 200,
        "output_tokens": 20,
        "cache_read_tokens": 160,
        "cache_write_tokens": 40,
        "model_calls": 2,
    }


async def test_workflows_set_aside_stay_available(inventory):
    agent, _ = agent_with(
        inventory,
        calls(("upscale_image", {"image": "a.png"})),
        answer("Feito."),
        calls(("new_workflow", {})),
        calls(("trim_video", {"video": "clip.mp4"})),
        answer("Outro workflow."),
    )
    first = await agent.send("amplie a imagem", files={"a.png": png()})

    await agent.send("agora corte o vídeo", files={"clip.mp4": MP4})

    assert agent.previous_workflows == [first.workflow]
    assert "Video Slice" in {node["class_type"] for node in agent.workflow.values()}


async def test_a_refusal_is_reported(inventory):
    agent, _ = agent_with(inventory, answer("", stop_reason="refusal"))

    reply = await agent.send("pedido recusado")

    assert reply.status == "refused"


def test_the_api_key_comes_from_the_env_file(tmp_path, monkeypatch, inventory):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"

    with pytest.raises(SettingsError):
        WorkflowAgent(inventory, env_file=env)

    env.write_text("ANTHROPIC_API_KEY: sk-test\n", encoding="utf-8")
    WorkflowAgent(inventory, env_file=env)


async def test_without_models_only_the_media_pipelines_are_offered():
    model = ScriptedChatModel(replies=[answer("Ok.")], profile=PROFILE)
    agent = WorkflowAgent(ServerInventory(), chat_model=model)

    await agent.send("oi")

    assert "text_to_image" not in model.tool_names
    assert "upscale_image" in model.tool_names
    assert "no model the generation pipelines know" in model.received[0][0].content
