from comfy_ui_api_workflow.nodes.image import EmptyImage, LoadImage, PreviewImage, SaveImage

from agent_comfy_ui_api_workflow._impl.validation import validate_workflow
from builders import text_to_image
from conftest import SDXL
from media import png


def check(session, data=None, strict_nodes=None):
    return validate_workflow(
        session.snapshot() if data is None else data,
        catalog=session.catalog,
        inventory=session.inventory,
        files=session.files,
        strict_nodes=strict_nodes,
    )


def test_a_complete_workflow_is_valid(session):
    text_to_image(session.workflow)

    report = check(session)

    assert report.ok
    assert report.problems == ()
    assert report.text() == "Valid: ready to run."


def test_an_empty_workflow_is_not_valid(session):
    report = check(session)

    assert not report.ok
    assert "The workflow is empty" in report.text()


def test_a_workflow_without_an_output_node_is_not_valid(session):
    session.workflow.add_node(EmptyImage, width=64, height=64)

    report = check(session)

    assert not report.ok
    assert "no output node" in report.text()


def test_a_node_that_reaches_no_output_blocks_only_when_the_agent_made_it(session):
    text_to_image(session.workflow)
    session.workflow.add_node(EmptyImage, width=64, height=64)

    assert not check(session).ok
    lenient = check(session, strict_nodes=set())
    assert lenient.ok
    (problem,) = lenient.problems
    assert problem.node_id == "EmptyImage_7"
    assert not problem.blocking


def test_a_model_that_is_not_on_the_server_blocks(session):
    text_to_image(session.workflow, checkpoint="flux1-dev.safetensors")

    report = check(session)

    assert not report.ok
    assert "'flux1-dev.safetensors'" in report.text()
    assert SDXL in report.text()


def test_a_file_the_user_did_not_attach_blocks_until_it_is_attached(session):
    image = session.workflow.add_node(LoadImage, image="me.png")
    session.workflow.add_node(SaveImage, images=image.output["IMAGE"])

    report = check(session)
    assert not report.ok
    assert "'me.png'" in report.text()

    session.files.add("me.png", png())
    assert check(session).ok


def test_a_file_in_a_node_the_agent_did_not_touch_is_a_warning(session):
    image = session.workflow.add_node(LoadImage, image="me.png")
    session.workflow.add_node(SaveImage, images=image.output["IMAGE"])

    report = check(session, strict_nodes=set())

    assert report.ok
    assert "upload 'me.png'" in report.text()


def test_previews_alone_save_no_file(session):
    image = session.workflow.add_node(EmptyImage, width=64, height=64)
    session.workflow.add_node(PreviewImage, images=image.output["IMAGE"])

    report = check(session)

    assert report.ok
    assert "saves no file" in report.text()


def test_a_workflow_that_does_not_load_back_is_reported(session):
    report = check(session, data={"1": {"class_type": "Nope", "inputs": {}}})

    assert not report.ok
    assert "Nope" in report.text()
