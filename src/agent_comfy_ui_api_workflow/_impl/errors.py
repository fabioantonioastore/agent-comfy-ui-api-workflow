class AgentComfyUIError(Exception):
    pass


class SettingsError(AgentComfyUIError):
    def __init__(self, missing: list[str], source: str) -> None:
        super().__init__(missing, source)
        self.missing = missing
        self.source = source

    def __str__(self) -> str:
        names = ", ".join(self.missing)
        return f"Missing settings: {names}. Set them in the environment or in {self.source}."


class RunPodError(AgentComfyUIError):
    def __init__(self, reason: str, job_id: str | None = None) -> None:
        super().__init__(reason, job_id)
        self.reason = reason
        self.job_id = job_id

    def __str__(self) -> str:
        if self.job_id is None:
            return f"RunPod request failed: {self.reason}."
        return f"RunPod job {self.job_id!r} failed: {self.reason}."


class JobTimeoutError(RunPodError):
    pass


class InventoryError(AgentComfyUIError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def __str__(self) -> str:
        return f"Cannot read the server's model inventory: {self.reason}."


class InputFileError(AgentComfyUIError):
    def __init__(self, name: str, reason: str) -> None:
        super().__init__(name, reason)
        self.name = name
        self.reason = reason

    def __str__(self) -> str:
        return f"Cannot use the file {self.name!r}: {self.reason}."


# Raised by the agent's tools; the message goes back to the model, so it says how to fix
# the call.
class ToolInputError(AgentComfyUIError):
    pass
