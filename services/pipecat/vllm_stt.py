"""vLLM realtime speech-to-text for the Pipecat pipeline.

Pipecat's OpenAI realtime client speaks a different session dialect. This
adapter sends 16 kHz PCM and reads ``transcription.delta`` / ``transcription.done``.
"""

import base64
import json
import os
import re

import modal
from loguru import logger
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.utils.time import time_now_iso8601
from websockets.protocol import State

STT_APP_NAME = "qwen3-asr-vllm"
STT_SERVER_NAME = "VLLMServer"


def _stt_server() -> modal.Server:
    return modal.Server.from_name(
        STT_APP_NAME,
        STT_SERVER_NAME,
        environment_name="shababo-dev",
    )


def _ws_realtime_url(service_url: str) -> str:
    """Turn an http(s) service URL into the vLLM realtime websocket URL."""
    base = service_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base.removeprefix("https://")
    elif base.startswith("http://"):
        base = "ws://" + base.removeprefix("http://")
    elif not (base.startswith("wss://") or base.startswith("ws://")):
        raise RuntimeError(f"unsupported STT URL {service_url}")
    if not base.endswith("/v1/realtime"):
        base += "/v1/realtime"
    return base


def stt_ws_url() -> str:
    """Websocket URL for the qwen3-asr realtime transcription session.

    ``STT_BASE_URL`` skips the Server lookup. An http(s) override is rewritten
    to ws(s) and ``/v1/realtime`` is appended when it is missing.
    """
    override = os.environ.get("STT_BASE_URL")
    if override:
        return _ws_realtime_url(override)
    service_url = _stt_server().get_url()
    if not service_url:
        raise RuntimeError(f"{STT_APP_NAME} {STT_SERVER_NAME} has no URL")
    return _ws_realtime_url(service_url)


# vLLM's realtime endpoint expects 16 kHz PCM. OpenAI's is 24 kHz.
_PCM_RATE = 16000
# Audio kept locally so the start of an utterance is not clipped. vLLM 0.32
# transcribes whatever has been appended, in 5 s slices, and a slice of
# silence comes back as "嗯".
_PREROLL_SECS = 0.5
# Qwen3-ASR writes "language Chinese<asr_text>…" around the words. This server
# streams that wrapper as transcript text. session.update language is ignored.
_QWEN_PREAMBLE = re.compile(r"language\s*[A-Za-z]*\s*<asr_text>", re.IGNORECASE)


class ModalQwen3ASRService(OpenAIRealtimeSTTService):
    """vLLM ``/v1/realtime`` transcription.

    Generation starts on ``input_audio_buffer.commit`` and a short utterance
    is only transcribed when that commit is ``final``. Audio is forwarded
    only while the user is speaking, plus a short preroll.
    """

    def __init__(self, *, api_key: str, base_url: str, model: str, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            turn_detection=False,
            settings=OpenAIRealtimeSTTService.Settings(model=model),
            **kwargs,
        )
        self._raw_transcript = ""
        self._last_final = ""
        self._speaking = False
        self._preroll = bytearray()
        self._utterance_bytes = 0

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            self._session_ready = False
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._websocket = await self._websocket_connect(
                uri=self._base_url,
                additional_headers=headers,
            )
            await self._call_event_handler("on_connected")
        except Exception as exc:
            await self.push_error(
                error_msg=f"Error connecting to vLLM Realtime STT: {exc}",
                exception=exc,
            )
            self._websocket = None

    async def _send_session_update(self):
        self._raw_transcript = ""
        await self._ws_send(
            {
                "type": "session.update",
                "model": self._settings.model,
                "language": "en",
            }
        )
        self._session_ready = True

    async def process_frame(self, frame, direction):
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            self._raw_transcript = ""
            self._last_final = ""
            preroll = bytes(self._preroll)
            self._preroll.clear()
            if preroll:
                await self._append_audio(preroll)
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False

    async def _send_audio(self, audio: bytes):
        audio = await self._resampler.resample(audio, self.sample_rate, _PCM_RATE)
        if not audio:
            return
        if not self._speaking:
            self._preroll.extend(audio)
            max_bytes = int(_PCM_RATE * 2 * _PREROLL_SECS)
            extra = len(self._preroll) - max_bytes
            if extra > 0:
                del self._preroll[:extra]
            return
        await self._append_audio(audio)

    async def _append_audio(self, audio: bytes):
        self._utterance_bytes += len(audio)
        await self._ws_send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("utf-8"),
            }
        )

    async def _commit_audio_buffer(self):
        seconds = self._utterance_bytes / (_PCM_RATE * 2)
        self._utterance_bytes = 0
        logger.info(f"stt commit {seconds:.2f}s")
        # Non-final starts generation. Final flushes a clip shorter than the
        # server's 5 s segment.
        await self._ws_send({"type": "input_audio_buffer.commit"})
        await self._ws_send({"type": "input_audio_buffer.commit", "final": True})

    async def _receive_messages(self):
        if not self._websocket:
            raise Exception("Websocket not connected")
        async for message in self._websocket:
            try:
                evt = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Failed to parse WebSocket message")
                continue
            evt_type = evt.get("type", "")
            if evt_type == "session.created":
                self._raw_transcript = ""
                await self._handle_session_created(evt)
            elif evt_type == "transcription.delta":
                self._raw_transcript += evt.get("delta", "")
                speech = _qwen_speech(self._raw_transcript)
                if speech:
                    await self._push_transcript(speech, final=False)
            elif evt_type == "transcription.done":
                raw = evt.get("text") or self._raw_transcript
                self._raw_transcript = ""
                speech = _qwen_speech(raw)
                if speech and speech != self._last_final:
                    self._last_final = speech
                    await self._push_transcript(speech, final=True)
            elif evt_type == "error":
                await self._handle_error(evt)
            else:
                logger.trace(f"Unhandled vLLM realtime event: {evt_type}")

    async def _handle_error(self, evt: dict):
        error = evt.get("error", {})
        if isinstance(error, str):
            error_msg = error
            error_code = evt.get("code", "")
        else:
            error_msg = error.get("message", "Unknown error")
            error_code = error.get("code", "")
        msg = f"vLLM Realtime STT error [{error_code}]: {error_msg}"
        await self.push_error(error_msg=msg)
        raise Exception(msg)

    async def _push_transcript(self, speech: str, *, final: bool):
        if final:
            logger.info(f"stt {speech}")
        frame = (
            TranscriptionFrame(speech, self._user_id, time_now_iso8601(), finalized=True)
            if final
            else InterimTranscriptionFrame(speech, self._user_id, time_now_iso8601())
        )
        await self.push_frame(frame)


def _qwen_speech(raw: str) -> str | None:
    """Drop Qwen's language wrapper and return the spoken text.

    ``None`` means the wrapper is still arriving and there is no speech yet.
    """
    if "<asr_text>" not in raw and raw.lstrip().lower().startswith("language"):
        return None
    return _QWEN_PREAMBLE.sub("", raw).replace("<asr_text>", "").strip()
