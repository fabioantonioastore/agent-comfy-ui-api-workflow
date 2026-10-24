import base64
import json

import httpx
import pytest

from agent_comfy_ui_api_workflow._impl.errors import JobTimeoutError, RunPodError
from agent_comfy_ui_api_workflow._impl.inventory import PLACEHOLDER
from agent_comfy_ui_api_workflow._impl.runpod import OutputFile, RunPodClient, endpoint_url
from test_inventory import LISTING, OBJECT_INFO


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class FakeEndpoint:
    """Answers like RunPod: /run queues the job, /status walks through `statuses`.

    A status can also be a response to send as it is, or an error to raise.
    """

    def __init__(self, statuses: list[dict | httpx.Response | Exception]) -> None:
        self.statuses = statuses
        self.requests: list[tuple[str, str, dict, dict | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, dict(request.headers), body))
        if request.url.path.endswith("/run"):
            return httpx.Response(200, json={"id": "job-1", "status": "IN_QUEUE"})
        if "/status/" in request.url.path:
            status = self.statuses.pop(0)
            if isinstance(status, Exception):
                raise status
            if isinstance(status, httpx.Response):
                return status
            return httpx.Response(200, json={"id": "job-1", **status})
        if "/cancel/" in request.url.path:
            return httpx.Response(200, json={"id": "job-1", "status": "CANCELLED"})
        return httpx.Response(404, text="not found")

    def paths(self) -> list[str]:
        return [path for _, path, _, _ in self.requests]


def client_for(endpoint: FakeEndpoint) -> RunPodClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
    return RunPodClient("abc123", "secret", http_client=http, poll_interval=0)


async def test_runs_a_job_with_files_and_decodes_its_outputs():
    endpoint = FakeEndpoint(
        [
            {"status": "IN_PROGRESS"},
            {
                "status": "COMPLETED",
                "executionTime": 1234,
                "delayTime": 56,
                "output": {
                    "images": [
                        {"filename": "out_00001_.png", "type": "base64", "data": b64(b"PNG")},
                        {"filename": "ignored.png", "type": "s3_url", "data": "https://x"},
                    ]
                },
            },
        ]
    )
    client = client_for(endpoint)

    result = await client.run({"1": {"class_type": "X", "inputs": {}}}, files={"me.png": b"img"})

    assert result.ok
    assert result.files == (OutputFile("out_00001_.png", b"PNG"),)
    assert (result.execution_ms, result.delay_ms) == (1234, 56)
    method, path, headers, body = endpoint.requests[0]
    assert (method, path) == ("POST", "/v2/abc123/run")
    assert headers["authorization"] == "Bearer secret"
    assert body["input"]["workflow"] == {"1": {"class_type": "X", "inputs": {}}}
    assert body["input"]["images"] == [{"name": "me.png", "image": b64(b"img")}]
    assert endpoint.paths()[1:] == ["/v2/abc123/status/job-1"] * 2


async def test_output_file_names_keep_only_the_base_name():
    output = {"images": [{"filename": "../../etc/out.png", "type": "base64", "data": b64(b"x")}]}
    client = client_for(FakeEndpoint([{"status": "COMPLETED", "output": output}]))

    result = await client.run({})

    assert result.files == (OutputFile("out.png", b"x"),)


async def test_a_failed_job_comes_back_with_its_error():
    client = client_for(FakeEndpoint([{"status": "FAILED", "error": "boom"}]))

    result = await client.run({})

    assert not result.ok
    assert result.status == "FAILED"
    assert result.error == "boom"


async def test_a_completed_job_with_node_errors_is_not_ok():
    # worker-comfyui reports a job COMPLETED when it has any file to return, even when a
    # node failed; the files are then only the inputs it loaded.
    errors = [
        "Workflow execution error: Node Type: SaveVideo, Node ID: SaveVideo_3, Message: boom\n",
        "Workflow execution error: Node Type: SaveImage, Node ID: SaveImage_9, Message: bang\n",
    ]
    output = {"errors": errors, "images": [{"filename": "in.mp4", "type": "base64", "data": b64(b"mp4")}]}
    client = client_for(FakeEndpoint([{"status": "COMPLETED", "output": output}]))

    result = await client.run({})

    assert not result.ok
    assert result.status == "COMPLETED"
    assert result.error == (
        "Workflow execution error: Node Type: SaveVideo, Node ID: SaveVideo_3, Message: boom\n"
        "Workflow execution error: Node Type: SaveImage, Node ID: SaveImage_9, Message: bang"
    )
    assert result.files == (OutputFile("in.mp4", b"mp4"),)


async def test_a_job_that_runs_too_long_is_cancelled():
    endpoint = FakeEndpoint([{"status": "IN_PROGRESS"}] * 5)
    client = client_for(endpoint)

    with pytest.raises(JobTimeoutError) as error:
        await client.run({}, timeout=0)

    assert error.value.job_id == "job-1"
    assert "/v2/abc123/cancel/job-1" in endpoint.paths()


async def test_a_status_request_that_fails_is_asked_again():
    # RunPod's API fails now and then while the job runs on.
    endpoint = FakeEndpoint(
        [
            {"status": "IN_PROGRESS"},
            httpx.Response(500, json={"status": 500, "title": "Internal Server Error"}),
            httpx.ConnectError("connection reset"),
            httpx.Response(429, text="slow down"),
            {"status": "COMPLETED", "output": {"images": []}},
        ]
    )
    client = client_for(endpoint)

    result = await client.run({})

    assert result.ok
    assert endpoint.paths().count("/v2/abc123/status/job-1") == 5


async def test_a_status_that_keeps_failing_ends_in_the_timeout_with_the_reason():
    endpoint = FakeEndpoint([httpx.Response(503, text="unavailable")] * 10_000)
    http = httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
    client = RunPodClient("abc123", "secret", http_client=http, poll_interval=0.001)

    with pytest.raises(JobTimeoutError) as error:
        await client.run({}, timeout=0.05)

    assert "HTTP 503" in str(error.value)
    assert "/v2/abc123/cancel/job-1" in endpoint.paths()


async def test_a_timeout_says_when_cancelling_failed_too():
    # With RunPod's API down, the job can be neither followed nor cancelled.
    endpoint = FakeEndpoint([httpx.Response(503, text="unavailable")] * 10_000)

    def down(request: httpx.Request) -> httpx.Response:
        if "/cancel/" in request.url.path:
            return httpx.Response(503, text="unavailable")
        return endpoint(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(down))
    client = RunPodClient("abc123", "secret", http_client=http, poll_interval=0.001)

    with pytest.raises(JobTimeoutError) as error:
        await client.run({}, timeout=0.05)

    assert error.value.job_id == "job-1"
    assert "cancelling it failed too" in str(error.value)


async def test_a_refused_status_request_is_not_asked_again():
    endpoint = FakeEndpoint([httpx.Response(404, text="no such job"), {"status": "COMPLETED"}])
    client = client_for(endpoint)

    with pytest.raises(RunPodError) as error:
        await client.run({})

    assert "HTTP 404" in str(error.value)
    assert endpoint.paths().count("/v2/abc123/status/job-1") == 1


async def test_http_errors_raise_a_runpod_error():
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    http = httpx.AsyncClient(transport=httpx.MockTransport(unauthorized))
    client = RunPodClient("abc123", "wrong", http_client=http, poll_interval=0)

    with pytest.raises(RunPodError) as error:
        await client.run({})

    assert "HTTP 401" in str(error.value)


async def test_discovers_the_inventory_from_object_info():
    endpoint = FakeEndpoint([{"status": "COMPLETED", "output": {"object_info": OBJECT_INFO}}])
    client = client_for(endpoint)

    inventory = await client.discover_inventory()

    assert inventory.folder("upscale_models") == ("RealESRGAN_x4plus.safetensors",)
    assert endpoint.requests[0][3] == {"input": {"object_info": True}}
    assert len(endpoint.requests) == 2


async def test_object_info_returns_the_servers_node_definitions():
    endpoint = FakeEndpoint([{"status": "COMPLETED", "output": {"object_info": OBJECT_INFO}}])
    client = client_for(endpoint)

    info = await client.object_info()

    assert info == OBJECT_INFO
    assert endpoint.requests[0][3] == {"input": {"object_info": True}}


async def test_object_info_from_a_worker_without_it_is_an_error():
    client = client_for(FakeEndpoint([{"status": "FAILED", "error": "Missing 'workflow' parameter"}]))

    with pytest.raises(RunPodError) as error:
        await client.object_info()

    assert "object_info" in str(error.value)
    assert "Missing 'workflow' parameter" in str(error.value)


async def test_falls_back_to_the_probe_on_a_worker_without_object_info():
    endpoint = FakeEndpoint(
        [{"status": "FAILED", "error": "Missing 'workflow' parameter"}, {"status": "FAILED", "error": LISTING}]
    )
    client = client_for(endpoint)

    inventory = await client.discover_inventory()

    assert inventory.folder("checkpoints") == ("sd_xl_base_1.0.safetensors",)
    sent = endpoint.requests[2][3]["input"]["workflow"]
    assert PLACEHOLDER in json.dumps(sent)


def test_endpoint_url_accepts_an_id_or_a_url():
    assert endpoint_url("abc") == "https://api.runpod.ai/v2/abc"
    assert endpoint_url("https://api.runpod.ai/v2/abc/") == "https://api.runpod.ai/v2/abc"


def test_from_env_reads_the_endpoint_and_key(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("RUNPOD_API_KEY: k\nRUNPOD_ENDPOINT_ID: https://api.runpod.ai/v2/xyz", encoding="utf-8")

    client = RunPodClient.from_env(env)

    assert client.url == "https://api.runpod.ai/v2/xyz"
