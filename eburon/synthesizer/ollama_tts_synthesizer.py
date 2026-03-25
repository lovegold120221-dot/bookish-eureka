import asyncio
import base64
import copy
import json
import logging
import os
import time
import uuid
from collections import deque

from .base_synthesizer import BaseSynthesizer
from eburon.helpers.logger_config import configure_logger
from eburon.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)


class OllamaTTSSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        model="llama3.2:latest",
        voice=None,
        base_url=None,
        api_key=None,
        stream=True,
        buffer_size=40,
        sampling_rate=22050,
        use_mulaw=True,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance", None), stream)

        self.model = model
        self.voice = voice or "nova"
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        self.sampling_rate = int(sampling_rate)
        self.use_mulaw = use_mulaw
        self.buffer_size = buffer_size

        self.sample_rate = self.sampling_rate
        self.audio_format = "mulaw" if use_mulaw else "wav"

        self.first_chunk_generated = False
        self.last_text_sent = False
        self.meta_info = None
        self.conversation_ended = False
        self.connection_error = None
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.current_text = ""

        self.sender_task = None
        self.internal_queue = asyncio.Queue()
        self.text_queue = deque()
        self.ws_send_time = None

        self.synthesized_characters = 0

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        try:
            if self.conversation_ended:
                return

            if not self.should_synthesize_response(sequence_id):
                logger.info(f"Not synthesizing text as sequence_id ({sequence_id}) is not in current task")
                await self.flush_synthesizer_stream()
                return

            if text != "":
                self.ws_send_time = time.perf_counter()

                import aiohttp

                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                payload = {
                    "model": self.model,
                    "text": text,
                    "voice": self.voice,
                }

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/api/audio/speech", json=payload, headers=headers
                        ) as response:
                            if response.status == 200:
                                audio_data = await response.read()
                                if audio_data:
                                    self.last_text_sent = True
                                    yield audio_data, text
                            else:
                                error_text = await response.text()
                                logger.error(f"OllamaTTS API error: {response.status} - {error_text}")
                                self.connection_error = f"API error: {response.status}"

                except Exception as e:
                    logger.error(f"OllamaTTS synthesis error: {e}")
                    self.connection_error = str(e)

            if end_of_llm_stream:
                self.last_text_sent = True

        except asyncio.CancelledError:
            logger.info("OllamaTTS sender task cancelled")
        except Exception as e:
            logger.error(f"Error in OllamaTTS sender: {e}")
            self.connection_error = str(e)

    async def generate(self):
        try:
            if self.stream:
                while True:
                    message = await self.internal_queue.get()

                    if message.get("end_of_stream"):
                        break

                    meta_info = message.get("meta_info", {})
                    text = message.get("data", "")
                    sequence_id = meta_info.get("sequence_id")

                    self.current_text = text
                    self.synthesized_characters += len(text) if text else 0

                    try:
                        if self.current_turn_start_time is None:
                            self.current_turn_start_time = time.perf_counter()
                            self.current_turn_id = meta_info.get("turn_id") or meta_info.get("sequence_id")
                    except Exception:
                        pass

                    async for audio_chunk, text_synthesized in self.sender(
                        text, sequence_id, end_of_llm_stream=meta_info.get("end_of_llm_stream", False)
                    ):
                        if self.connection_error:
                            raise Exception(self.connection_error)

                        meta_info_copy = copy.deepcopy(meta_info)

                        if not self.first_chunk_generated:
                            meta_info_copy["is_first_chunk"] = True
                            self.first_chunk_generated = True
                        else:
                            meta_info_copy["is_first_chunk"] = False

                        if self.use_mulaw:
                            meta_info_copy["format"] = "mulaw"
                        else:
                            meta_info_copy["format"] = "wav"

                        meta_info_copy["text_synthesized"] = text_synthesized
                        meta_info_copy["mark_id"] = str(uuid.uuid4())

                        yield create_ws_data_packet(audio_chunk, meta_info_copy)

            else:
                while True:
                    message = await self.internal_queue.get()

                    if message.get("end_of_stream"):
                        break

                    meta_info = message.get("meta_info", {})
                    text = message.get("data", "")

                    self.synthesized_characters += len(text) if text else 0

                    import aiohttp

                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"

                    payload = {
                        "model": self.model,
                        "text": text,
                        "voice": self.voice,
                    }

                    audio = b""
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                f"{self.base_url}/api/audio/speech", json=payload, headers=headers
                            ) as response:
                                if response.status == 200:
                                    audio = await response.read()
                    except Exception as e:
                        logger.error(f"OllamaTTS HTTP synthesis error: {e}")

                    meta_info_copy = copy.deepcopy(meta_info)
                    meta_info_copy["is_first_chunk"] = True
                    meta_info_copy["end_of_synthesizer_stream"] = True

                    if self.use_mulaw:
                        meta_info_copy["format"] = "mulaw"
                    else:
                        meta_info_copy["format"] = "wav"

                    yield create_ws_data_packet(audio, meta_info_copy)

        except Exception as e:
            logger.error(f"Error in OllamaTTS generate: {e}")
            raise

    async def push(self, message):
        if self.stream:
            self.internal_queue.put_nowait(message)
        else:
            self.internal_queue.put_nowait(message)

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def supports_websocket(self):
        return False

    async def flush_synthesizer_stream(self):
        self.clear_internal_queue()

    def get_engine(self):
        return f"ollama-tts-{self.model}"

    async def cleanup(self):
        self.conversation_ended = True
        logger.info("Cleaning up OllamaTTS synthesizer")
        if self.sender_task:
            self.sender_task.cancel()
            try:
                await self.sender_task
            except asyncio.CancelledError:
                pass
