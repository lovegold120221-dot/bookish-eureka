import asyncio
import logging
import numpy as np
import os
import time
from dotenv import load_dotenv

from .base_transcriber import BaseTranscriber
from eburon.helpers.logger_config import configure_logger
from eburon.helpers.utils import create_ws_data_packet, timestamp_ms

logger = configure_logger(__name__)
load_dotenv()


class WhisperTranscriber(BaseTranscriber):
    def __init__(
        self,
        input_queue=None,
        model="base",
        language=None,
        device=None,
        compute_type="float16",
        streaming=True,
        **kwargs,
    ):
        super().__init__(input_queue)
        self.model_name = model
        self.language = language
        self.streaming = streaming
        self.device = device or ("cuda" if os.getenv("CUDA_AVAILABLE") else "cpu")
        self.compute_type = compute_type
        self.transcriber_output_queue = kwargs.get("output_queue")
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.is_speaking = False
        self.last_speech_time = None
        self.speech_timeout = 2.0
        self.model = None
        self.transcription_task = None
        self.sender_task = None
        self._initialized = False
        self.connection_on = True

    async def initialize(self):
        if self._initialized:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model '{self.model_name}' on {self.device} with {self.compute_type}")

            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._initialized = True
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.warning("faster-whisper not installed, trying torch with transformers")
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                model_id = f"openai/whisper-{self.model_name}"
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                )
                processor = AutoProcessor.from_pretrained(model_id)

                self.pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                self._initialized = True
                logger.info("Transformers Whisper pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

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
            logger.info("Whisper sender task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Whisper sender: {e}")

    async def receiver(self):
        try:
            min_audio_duration = 0.5
            speech_pad_duration = 0.5

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
                    audio_data = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_buffer = bytearray()

                    if len(audio_data) < self.sample_rate * min_audio_duration:
                        continue

                    self.current_request_id = self.generate_request_id()
                    self.meta_info = {"request_id": self.current_request_id}

                    try:
                        if hasattr(self, "pipeline"):
                            result = self.pipeline(
                                {"sampling_rate": self.sample_rate, "raw": audio_data},
                                generate_kwargs={"language": self.language} if self.language else {},
                            )
                            transcript = result["text"].strip()
                        else:
                            segments, _ = self.model.transcribe(
                                audio_data,
                                language=self.language,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=500),
                            )
                            transcript = " ".join([segment.text for segment in segments]).strip()

                        if transcript:
                            logger.info(f"Whisper transcription: {transcript}")
                            data = {"type": "transcript", "content": transcript}
                            yield create_ws_data_packet(data, self.meta_info)

                    except Exception as e:
                        logger.error(f"Whisper transcription error: {e}")

        except asyncio.CancelledError:
            logger.info("Whisper receiver task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Whisper receiver: {e}")

    async def run(self):
        await self.initialize()
        try:
            self.transcription_task = asyncio.create_task(self._transcribe())
        except Exception as e:
            logger.error(f"Failed to start Whisper transcription: {e}")

    async def _transcribe(self):
        try:
            sender_task = asyncio.create_task(self.sender_stream())

            async for message in self.receiver():
                if self.connection_on:
                    await self.push_to_transcriber_queue(message)

        except asyncio.CancelledError:
            logger.info("Whisper transcription task cancelled")
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
        finally:
            if hasattr(self, "sender_task") and self.sender_task:
                self.sender_task.cancel()

    async def push_to_transcriber_queue(self, data_packet):
        if self.transcriber_output_queue:
            await self.transcriber_output_queue.put(data_packet)

    async def cleanup(self):
        self.connection_on = False
        logger.info("Cleaning up Whisper transcriber")
        if self.transcription_task:
            self.transcription_task.cancel()
            try:
                await self.transcription_task
            except asyncio.CancelledError:
                pass
        self.audio_buffer = bytearray()
