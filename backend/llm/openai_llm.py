"""OpenAI LLM setup and LangSmith tracing configuration."""

import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langsmith import traceable, trace
from langsmith.client import RUN_TYPE_T

load_dotenv()

#  OpenAI
_openai_key_value = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = SecretStr(_openai_key_value) if _openai_key_value else None

#  LangSmith 
# LangSmith auto-reads these env vars:
#   LANGSMITH_API_KEY
#   LANGSMITH_ENDPOINT  (or LANGSMITH_API_URL)
#   LANGSMITH_PROJECT
#   LANGSMITH_TRACING   (set to "true" to enable)

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")


def langsmith_trace(
    name: str,
    run_type: RUN_TYPE_T = "chain",
    inputs: dict | None = None,
    extra: dict | None = None,
):
    """
    Return a LangSmith trace context manager when tracing is enabled,
    or a no-op context manager when it is disabled.

    Usage:
        with langsmith_trace("MyAgent.run", run_type="llm", inputs={...}) as run:
            response = model.invoke(...)
            run.end(outputs={"result": response})
    """
    if not LANGSMITH_TRACING:
        return _NoopTrace()

    return trace(
        name,
        run_type=run_type,
        inputs=inputs or {},
        extra=extra or {},
        project_name=LANGSMITH_PROJECT,
    )


class _NoopTrace:
    """Silent no-op used when LangSmith tracing is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def end(self, outputs: dict | None = None):
        pass


# ── Convenience decorator (preferred for simple agent methods) ─────────────────
# Use @traceable instead of the context manager when you don't need
# to customise the run object mid-flight.
#
# Example:
#   from llm.openai_llm import traceable
#
#   @traceable(run_type="llm", name="ResearchAgent.generate")
#   def generate(question: str, context: str) -> str:
#       ...
#
# `traceable` is re-exported from langsmith so callers only need one import.
__all__ = ["OPENAI_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "langsmith_trace", "traceable"]
