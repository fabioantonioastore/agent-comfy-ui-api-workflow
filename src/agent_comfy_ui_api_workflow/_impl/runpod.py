import asyncio
import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from time import monotonic
from typing import Any

import httpx

from .env import load_settings
from .errors import JobTimeoutError, RunPodError
from .inventory import ServerInventory, parse_listing, parse_object_info, probe_workflow

SETTINGS = ("RUNPOD_ENDPOINT_ID", "RUNPOD_API_KEY")
_RUNNING = ("IN_QUEUE", "IN_PROGRESS")


class _TransientError(RunPodError):
    """A failure of RunPod's API that asking again can get past: no reply, HTTP 429 or 5xx."""


def endpoint_url(endpoint: str) -> str:
    if not endpoint.startswith("http"):
        endpoint = f"https://api.runpod.ai/v2/{endpoint}"
    return endpoint.rstrip("/")


@dataclass(frozen=True)
class OutputFile:
    filename: str
    data: bytes


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: str
    output: Any
    error: str | None
    files: tuple[OutputFile, ...]
    execution_ms: int | None = None
    delay_ms: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "COMPLETED" and self.error is None


def _files(output: Any) -> tuple[OutputFile, ...]:
    if not isinstance(output, dict):
        return ()
    files: list[OutputFile] = []
    for item in output.get("images") or []:
        if not isinstance(item, dict) or item.get("type") != "base64":
            continue
        # Only the base name, so a name from the server cannot point outside a folder.
        name = PurePath(str(item.get("filename") or "output")).name
        files.append(OutputFile(name, base64.b64decode(item.get("data") or "")))
    return tuple(files)


def _node_errors(output: Any) -> str | None:
    # worker-comfyui reports a job COMPLETED when it has any file to return, and lists
    # the nodes that failed here.
    errors = output.get("errors") if isinstance(output, dict) else None
    return "\n".join(str(error).strip() for error in errors or []) or None


def _result(job: dict[str, Any]) -> JobResult:
    error = job.get("error")
    return JobResult(
        job_id=str(job.get("id")),
        status=str(job.get("status")),
        output=job.get("output"),
        error=_node_errors(job.get("output")) if error is None else str(error),
        files=_files(job.get("output")),
        execution_ms=job.get("executionTime"),
        delay_ms=job.get("delayTime"),
    )


def _object_info(result: JobResult) -> dict[str, Any] | None:
    info = result.output.get("object_info") if isinstance(result.output, dict) else None
    return info if result.ok and isinstance(info, dict) else None


class RunPodClient:
    """Runs API-format workflows on a RunPod serverless endpoint that serves ComfyUI."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        poll_interval: float = 2.0,
        request_timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = endpoint_url(endpoint)
        self.__api_key = api_key
        self.__poll_interval = poll_interval
        self.__http = http_client or httpx.AsyncClient(timeout=request_timeout)

    @classmethod
    def from_env(cls, env_file: str | Path = ".env", **options: Any) -> "RunPodClient":
        settings = load_settings(SETTINGS, env_file)
        return cls(settings["RUNPOD_ENDPOINT_ID"], settings["RUNPOD_API_KEY"], **options)

    async def __aenter__(self) -> "RunPodClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.__http.aclose()

    async def run(
        self,
        workflow: Mapping[str, Any],
        files: Mapping[str, bytes] | None = None,
        *,
        timeout: float = 600.0,
    ) -> JobResult:
        job_input: dict[str, Any] = {"workflow": dict(workflow)}
        if files:
            # worker-comfyui saves these in ComfyUI's input folder before the run.
            job_input["images"] = [
                {"name": name, "image": base64.b64encode(data).decode()}
                for name, data in files.items()
            ]
        return await self.__job(job_input, timeout)

    async def health(self) -> dict[str, Any]:
        return await self.__request("GET", "/health")

    async def object_info(self, *, timeout: float = 600.0) -> dict[str, Any]:
        """ComfyUI's /object_info: every node of the server, with its inputs and outputs.

        Only workers whose handler answers {"object_info": true} jobs have it.
        """
        result = await self.__job({"object_info": True}, timeout)
        info = _object_info(result)
        if info is None:
            reason = result.error or result.status
            raise RunPodError(f"the worker did not answer the object_info job ({reason})", result.job_id)
        return info

    async def discover_inventory(self, *, timeout: float = 600.0) -> ServerInventory:
        """Lists the server's model files: the options of its loaders in /object_info."""
        info = _object_info(await self.__job({"object_info": True}, timeout))
        if info is not None:
            return parse_object_info(info)
        # The official worker answers no object_info jobs. A probe that names a missing
        # file in each loader makes it list the folders, but only for the loaders its
        # handler checks, and without the options of V3 nodes.
        result = await self.run(probe_workflow(), timeout=timeout)
        reply = result.error if result.error is not None else json.dumps(result.output)
        return parse_listing(reply)

    async def __job(self, job_input: dict[str, Any], timeout: float) -> JobResult:
        job = await self.__request("POST", "/run", {"input": job_input})
        deadline = monotonic() + timeout
        failure: str | None = None
        while job.get("status") in _RUNNING:
            if monotonic() >= deadline:
                job_id = str(job["id"])
                reason = f"it did not finish in {timeout:g} s"
                if failure is not None:
                    reason += f" (its status last failed: {failure})"
                problem = await self.__cancel(job_id)
                reason += " and was cancelled" if problem is None else f"; {problem}"
                raise JobTimeoutError(reason, job_id)
            await asyncio.sleep(self.__poll_interval)
            try:
                job = await self.__request("GET", f"/status/{job['id']}")
                failure = None
            except _TransientError as error:
                # RunPod's API fails now and then while the job runs on: ask again.
                failure = error.reason
        return _result(job)

    async def __cancel(self, job_id: str) -> str | None:
        try:
            await self.__request("POST", f"/cancel/{job_id}")
        except RunPodError as error:
            return f"cancelling it failed too ({error.reason})"
        return None

    async def __request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = await self.__http.request(
                method,
                f"{self.url}{path}",
                json=body,
                headers={"Authorization": f"Bearer {self.__api_key}"},
            )
        except httpx.HTTPError as error:
            raise _TransientError(f"{method} {path} got no reply ({error})") from error
        if response.status_code >= 400:
            transient = response.status_code == 429 or response.status_code >= 500
            raise (_TransientError if transient else RunPodError)(
                f"{method} {path} returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()
