import aiohttp
import os
from collections import deque
from .base_synthesizer import BaseSynthesizer
from eburon.helpers.logger_config import configure_logger
from eburon.helpers.utils import create_ws_data_packet, resample, convert_audio_to_wav

logger = configure_logger(__name__)


class EburonEdgeSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice="ep_001",
        model="nike",
        language="en",
        sampling_rate=24000,
        stream=True,
        buffer_size=400,
        base_url="http://localhost:17493",
        stream_in_chunks=True,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance", None), stream)
        self.api_key = kwargs.get("synthesizer_key", os.getenv("EBURON_EDGE_API_KEY", ""))
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.model = model
        self.stream = stream
        self.buffer_size = buffer_size
        self.sampling_rate = int(sampling_rate)
        self.language = language
        self.stream_in_chunks = stream_in_chunks

        self.first_chunk_generated = False
        self.last_text_sent = False
        self.meta_info = None
        self.conversation_ended = False
        self.connection_error = None
        self.current_turn_start_time = None
        self.current_turn_id = None
        self.text_queue = deque()
        self.current_text = ""

        self.sender_task = None
        self._initialized = False

    def get_engine(self):
        return self.model

    async def __generate_http(self, text):
        url = f"{self.base_url}/generate"
        payload = {"text": text, "engine": self.model, "profile_id": self.voice, "params": {}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        return audio_data
                    else:
                        error_text = await response.text()
                        logger.error(f"Eburon Edge API error {response.status}: {error_text}")
                        raise Exception(f"API error {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")
            raise

    async def synthesize(self, text):
        audio = await self.__generate_http(text)
        return audio

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                meta_info["text"] = text

                if not self.should_synthesize_response(meta_info.get("sequence_id")):
                    logger.info(f"Skipping TTS for sequence_id: {meta_info.get('sequence_id')}")
                    return

                try:
                    audio = await self.__generate_http(text)
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True

                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False

                    audio_wav = convert_audio_to_wav(audio, format="mp3")
                    audio_resampled = resample(audio_wav, self.sampling_rate, format="wav")
                    yield create_ws_data_packet(audio_resampled, meta_info)

                except Exception as e:
                    logger.error(f"TTS generation error: {e}")
                    raise

        except Exception as e:
            logger.error(f"Error in Eburon Edge TTS generate: {e}")
            raise

    async def open_connection(self):
        pass

    async def push(self, message):
        logger.info(f"Pushed message to internal queue: {message}")
        self.internal_queue.put_nowait(message)
