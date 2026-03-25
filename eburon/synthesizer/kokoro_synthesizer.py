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


class KokoroSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice="af_sarah",
        model="kokoro-v0.19.onnx",
        voicepack_path=None,
        device=None,
        stream=True,
        buffer_size=40,
        sampling_rate=24000,
        use_mulaw=True,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance", None), stream)

        self.voice = voice
        self.model_name = model
        self.voicepack_path = voicepack_path or os.getenv("KOKORO_VOICEPACK_PATH", "./voices")
        self.device = device or ("cuda" if os.getenv("CUDA_AVAILABLE") else "cpu")
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
        self.text_queue = deque()
        self.ws_send_time = None

        self._initialized = False
        self._model = None
        self._voices = None

        self.synthesized_characters = 0

    async def initialize(self):
        if self._initialized:
            return

        try:
            from kokoro import KModel, VITS2
            import onnxruntime as ort

            logger.info(f"Loading Kokoro model '{self.model_name}' on {self.device}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            )

            self._model = KModel(
                self.model_name,
                voices=self.voicepack_path,
                device=self.device,
            )

            self._voices = self._model.voices if hasattr(self._model, "voices") else {}
            self._initialized = True
            logger.info("Kokoro model loaded successfully")

        except ImportError as e:
            logger.warning(f"Kokoro not installed: {e}. Will use HTTP fallback if available.")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load Kokoro model: {e}")
            self._initialized = True

    def synthesize_text(self, text):
        if not self._initialized:
            return b""

        try:
            if hasattr(self._model, "generate"):
                audio = self._model.generate(text, self.voice)
                return audio
        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}")

        return b""

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
                audio = self.synthesize_text(text)

                if len(audio) > 0:
                    self.last_text_sent = True
                    yield audio, text

            if end_of_llm_stream:
                self.last_text_sent = True

        except asyncio.CancelledError:
            logger.info("Kokoro sender task cancelled")
        except Exception as e:
            logger.error(f"Error in Kokoro sender: {e}")
            self.connection_error = str(e)

    async def generate(self):
        try:
            if self.stream:
                await self.initialize()

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
                await self.initialize()

                while True:
                    message = await self.internal_queue.get()

                    if message.get("end_of_stream"):
                        break

                    meta_info = message.get("meta_info", {})
                    text = message.get("data", "")

                    self.synthesized_characters += len(text) if text else 0

                    audio = self.synthesize_text(text)

                    meta_info_copy = copy.deepcopy(meta_info)
                    meta_info_copy["is_first_chunk"] = True
                    meta_info_copy["end_of_synthesizer_stream"] = True

                    if self.use_mulaw:
                        meta_info_copy["format"] = "mulaw"
                    else:
                        meta_info_copy["format"] = "wav"

                    yield create_ws_data_packet(audio, meta_info_copy)

        except Exception as e:
            logger.error(f"Error in Kokoro generate: {e}")
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
        return f"kokoro-{self.voice}"

    async def cleanup(self):
        self.conversation_ended = True
        logger.info("Cleaning up Kokoro synthesizer")
        if self.sender_task:
            self.sender_task.cancel()
            try:
                await self.sender_task
            except asyncio.CancelledError:
                pass
