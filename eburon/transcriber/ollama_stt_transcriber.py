import asyncio
import json
import logging
import os
import time
from dotenv import load_dotenv

from .base_transcriber import BaseTranscriber
from eburon.helpers.logger_config import configure_logger
from eburon.helpers.utils import create_ws_data_packet, timestamp_ms

logger = configure_logger(__name__)
load_dotenv()


class OllamaSTTTranscriber(BaseTranscriber):
    def __init__(
        self, input_queue=None, model="whisper", language=None, base_url=None, api_key=None, streaming=True, **kwargs
    ):
        super().__init__(input_queue)
        self.model = model
        self.language = language
        self.streaming = streaming
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        self.transcriber_output_queue = kwargs.get("output_queue")
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.last_speech_time = None
        self.speech_timeout = 2.0
        self.sample_rate = 16000
        self.transcription_task = None
        self.sender_task = None
        self.connection_on = True
        self.current_request_id = None
        self.meta_info = None

    async def sender_stream(self):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()

                if "eos" in ws_data_packet.get("meta_info", {}) and ws_data_packet["meta_info"]["eos"]:
                    break

                audio_data = ws_data_packet.get("data")
                if audio_data:
                    self.audio_buffer.extend(audio_data)

                self.last_speech_time = time.time()

        except asyncio.CancelledError:
            logger.info("OllamaSTT sender task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in OllamaSTT sender: {e}")

    async def receiver(self):
        import aiohttp

        min_audio_duration = 1.0

        while True:
            await asyncio.sleep(0.1)

            if len(self.audio_buffer) < int(self.sample_rate * min_audio_duration * 2):
                continue

            current_time = time.time()
            if (
                self.last_speech_time
                and current_time - self.last_speech_time > self.speech_timeout
                and len(self.audio_buffer) > 0
            ):
                audio_data = bytes(self.audio_buffer)
                self.audio_buffer = bytearray()

                if len(audio_data) < int(self.sample_rate * min_audio_duration * 2):
                    continue

                self.current_request_id = self.generate_request_id()
                self.meta_info = {"request_id": self.current_request_id}

                try:
                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"

                    import base64

                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                    payload = {
                        "model": self.model,
                        "audio": audio_b64,
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/api/audio/transcriptions", json=payload, headers=headers
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                transcript = result.get("text", "").strip()

                                if transcript:
                                    logger.info(f"OllamaSTT transcription: {transcript}")
                                    data = {"type": "transcript", "content": transcript}
                                    yield create_ws_data_packet(data, self.meta_info)
                            else:
                                error_text = await response.text()
                                logger.error(f"OllamaSTT API error: {response.status} - {error_text}")

                except Exception as e:
                    logger.error(f"OllamaSTT transcription error: {e}")

    async def run(self):
        try:
            self.transcription_task = asyncio.create_task(self._transcribe())
        except Exception as e:
            logger.error(f"Failed to start OllamaSTT transcription: {e}")

    async def _transcribe(self):
        try:
            sender_task = asyncio.create_task(self.sender_stream())

            async for message in self.receiver():
                if self.connection_on:
                    await self.push_to_transcriber_queue(message)

        except asyncio.CancelledError:
            logger.info("OllamaSTT transcription task cancelled")
        except Exception as e:
            logger.error(f"OllamaSTT transcription error: {e}")
        finally:
            if hasattr(self, "sender_task") and self.sender_task:
                self.sender_task.cancel()

    async def push_to_transcriber_queue(self, data_packet):
        if self.transcriber_output_queue:
            await self.transcriber_output_queue.put(data_packet)

    async def cleanup(self):
        self.connection_on = False
        logger.info("Cleaning up OllamaSTT transcriber")
        if self.transcription_task:
            self.transcription_task.cancel()
            try:
                await self.transcription_task
            except asyncio.CancelledError:
                pass
        self.audio_buffer = bytearray()
