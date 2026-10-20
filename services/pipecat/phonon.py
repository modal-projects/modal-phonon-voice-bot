"""Phonon text-to-speech for the Pipecat pipeline.

The engine is loaded once on the bot container. This service only turns text
into PCM and reports time-to-first-byte.
"""

import asyncio
import os

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

VOICE = os.environ.get("PHONON_VOICE", "default")


class PhononTTSService(TTSService):
    """Speak with a Phonon engine that is already loaded in this process."""

    def __init__(self, *, engine, voice: str = VOICE, **kwargs):
        super().__init__(
            sample_rate=engine.sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(
                model=None,
                voice=voice,
                language=Language.EN,
            ),
            **kwargs,
        )
        self._engine = engine

    def can_generate_metrics(self) -> bool:
        """Report time to first audio byte, the same way other TTS services do."""
        return True

    async def run_tts(self, text: str, context_id: str):
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        stream = self._engine.stream(text, voice=self._settings.voice or VOICE)

        def produce() -> None:
            try:
                for chunk in stream:
                    pcm = _f32_to_s16(chunk)
                    if pcm:
                        _put(loop, chunks, pcm)
            except Exception as exc:
                _put(loop, chunks, exc)
            else:
                _put(loop, chunks, None)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await chunks.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield TTSAudioRawFrame(item, self.sample_rate, 1, context_id=context_id)
        finally:
            stream.close()
            await producer


def _put(loop, chunks, item) -> None:
    asyncio.run_coroutine_threadsafe(chunks.put(item), loop).result()


def _f32_to_s16(chunk) -> bytes:
    import numpy as np

    samples = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()
