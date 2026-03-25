import os
import json
import time
import logging
import asyncio
from dotenv import load_dotenv

from eburon.constants import DEFAULT_LANGUAGE_CODE
from eburon.enums import LogComponent, LogDirection
from eburon.helpers.utils import convert_to_request_log, now_ms
from .llm import BaseLLM
from .tool_call_accumulator import ToolCallAccumulator
from .types import LLMStreamChunk, LatencyData
from eburon.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model,
        max_tokens=30,
        buffer_size=40,
        temperature=0.0,
        language=DEFAULT_LANGUAGE_CODE,
        base_url=None,
        **kwargs,
    ):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        self.started_streaming = False
        self.language = language
        self.run_id = kwargs.get("run_id", None)

        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_key = kwargs.get("llm_key", os.getenv("OLLAMA_API_KEY", "ollama"))

        self.model_args = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "options": {}}

        self.custom_tools = kwargs.get("api_tools", None)
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools.get("tools_params", {})
            self.tools = self.custom_tools.get("tools", None)
        else:
            self.trigger_function_call = False

    async def generate_stream(self, messages, synthesize=True, meta_info=None, tool_choice=None):
        if not messages or len(messages) == 0:
            raise Exception("No messages provided")

        answer, buffer = "", ""
        first_token_time = None

        start_time = now_ms()
        latency_data = LatencyData(
            sequence_id=meta_info.get("sequence_id") if meta_info else None,
        )

        try:
            import aiohttp

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": self.model_args.get("temperature", 0.0),
                    "num_predict": self.model_args.get("max_tokens", 100),
                },
            }

            if self.trigger_function_call and self.tools:
                payload["tools"] = json.loads(self.tools) if isinstance(self.tools, str) else self.tools

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/chat", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")

                    async for line in response.content:
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        now = now_ms()
                        if not first_token_time:
                            first_token_time = now
                            self.started_streaming = True
                            latency_data = LatencyData(
                                sequence_id=meta_info.get("sequence_id"),
                                first_token_latency_ms=first_token_time - start_time,
                            )

                        if chunk_data.get("done"):
                            break

                        content = chunk_data.get("message", {}).get("content", "")
                        if content:
                            answer += content
                            buffer += content
                            if synthesize and len(buffer) >= self.buffer_size:
                                split = buffer.rsplit(" ", 1)
                                yield LLMStreamChunk(data=split[0], end_of_stream=False, latency=latency_data)
                                buffer = split[1] if len(split) > 1 else ""

                        if chunk_data.get("done"):
                            break

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

        if latency_data:
            latency_data.total_stream_duration_ms = now_ms() - start_time

        if synthesize and buffer.strip():
            yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
        elif not synthesize:
            yield LLMStreamChunk(data=answer, end_of_stream=True, latency=latency_data)

        self.started_streaming = False

    async def generate(self, messages, stream=False, request_json=False, meta_info=None, ret_metadata=False):
        text = ""

        try:
            import aiohttp

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.model_args.get("temperature", 0.0),
                    "num_predict": self.model_args.get("max_tokens", 100),
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/chat", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")

                    result = await response.json()
                    text = result.get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            raise

        if ret_metadata:
            return text, {}
        else:
            return text
