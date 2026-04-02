"""OpenAI LLM setup and API key configuration for the backend."""

import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY_VALUE = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = (
    SecretStr(OPENAI_API_KEY_VALUE) if OPENAI_API_KEY_VALUE is not None else None
)

model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini")
