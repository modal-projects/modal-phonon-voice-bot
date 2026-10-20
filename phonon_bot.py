"""Pipecat voice bot on a Modal CPU container, with Phonon TTS in-process.

A sticky session pins each call to one container. Small WebRTC signaling
(the SDP offer and ICE candidate patches) has to reach that same process,
because the peer connection lives in memory next to Phonon. ``modal deploy
phonon_bot.py`` also publishes ``site``, which serves ``client/`` and proxies
signaling. The LLM
is the phonellm-mini-alpha-2 Endpoint. STT is the qwen3-asr-vllm Server's
realtime websocket. Silero VAD and Smart Turn run in this same container.
"""

import os
import threading
from pathlib import Path

import modal

from services.phonon_tts import MODEL_REMOTE, load_engine, phonon_volume

app = modal.App("phonon-pipecat-bot")

PORT = 8000
VOICE = os.environ.get("PHONON_VOICE", "Freya")

MODAL_ENVIRONMENT = "shababo-dev"
# Empty by default: the ASR server does not require a bearer token.
STT_API_KEY = os.environ.get("STT_API_KEY", "")
STT_MODEL = os.environ.get("STT_MODEL", "Qwen/Qwen3-ASR-0.6B")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "dummy-llm")
LLM_MODEL = os.environ.get("LLM_MODEL", "phonellm-mini-alpha-2")
# Dedicated Endpoint phonellm-mini-alpha-2. Its backing app is named
# ep-<endpoint name> and the server class is Server.
LLM_APP_NAME = "ep-phonellm-mini-alpha-2"
LLM_SERVER_NAME = "Server"
# Comma-separated STUN/TURN URLs, or a JSON list of URL strings or
# {"urls", "username", "credential"} objects. The bot gathers candidates
# from these. A browser on another network usually also needs a TURN relay.
ICE_SERVERS = os.environ.get("ICE_SERVERS", "stun:stun.l.google.com:19302")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libsndfile1", "libglib2.0-0")
    .uv_pip_install(
        "pipecat-ai[webrtc,websocket]==1.13.0",
        "ptts==0.4.0rc3",
        "uvicorn",
    )
    .add_local_python_source("services")
)


def _llm_server() -> modal.Server:
    return modal.Server.from_name(
        LLM_APP_NAME,
        LLM_SERVER_NAME,
        environment_name=MODAL_ENVIRONMENT,
    )


def _openai_base_url(service_url: str | None) -> str:
    """Chat-completions base URL. ``LLM_BASE_URL`` skips the Server lookup."""
    override = os.environ.get("LLM_BASE_URL")
    if override:
        return override.rstrip("/")
    if not service_url:
        raise RuntimeError(f"{LLM_APP_NAME} Server has no URL")
    return service_url.rstrip("/") + "/v1"


def llm_base_url() -> str:
    """OpenAI-compatible base URL for the phonellm Endpoint."""
    if os.environ.get("LLM_BASE_URL"):
        return _openai_base_url(None)
    return _openai_base_url(_llm_server().get_url())


def ice_server_specs() -> list[str | dict]:
    """Parse ``ICE_SERVERS`` into URL strings or IceServer keyword dicts."""
    import json

    raw = ICE_SERVERS.strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [part.strip() for part in raw.split(",") if part.strip()]


async def run_bot(connection, engine, session_id: str, llm_url: str, stt_url: str) -> None:
    from loguru import logger
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from services.pipecat.phonellm import ModalPhoneLLMMiniService
    from services.pipecat.phonon import PhononTTSService
    from services.pipecat.vllm_stt import ModalQwen3ASRService
    from pipecat.turns.user_start import (
        TranscriptionUserTurnStartStrategy,
        VADUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.workers.runner import WorkerRunner

    transport = SmallWebRTCTransport(
        connection,
        TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    # Server-side VAD stays off. Silero commits each speech segment; Smart Turn
    # decides whether that segment ends the user's turn.
    stt = ModalQwen3ASRService(api_key=STT_API_KEY, base_url=stt_url, model=STT_MODEL)
    llm = ModalPhoneLLMMiniService(
        api_key=LLM_API_KEY,
        base_url=llm_url,
        settings=ModalPhoneLLMMiniService.Settings(
            model=LLM_MODEL,
            system_instruction=(
                "You are a concise voice assistant. Your replies are spoken "
                "aloud, so use short sentences and no markdown."
            ),
        ),
    )
    tts = PhononTTSService(engine=engine, voice=VOICE)
    context = LLMContext()
    user, assistant = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(),
                    TranscriptionUserTurnStartStrategy(),
                ],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(),
                    )
                ],
            ),
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user,
            llm,
            tts,
            transport.output(),
            assistant,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=engine.sample_rate,
            enable_metrics=True,
        ),
        conversation_id=session_id or None,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, webrtc_connection):
        logger.info(f"client connected session={session_id} pc={webrtc_connection.pc_id}")
        await worker.queue_frames(
            [TTSSpeakFrame("Hi, how can I help you?")]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, webrtc_connection):
        logger.info(f"client disconnected session={session_id} pc={webrtc_connection.pc_id}")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

with image.imports():
    import ptts
    import uvicorn
    from fastapi import BackgroundTasks, FastAPI, Request
    from pipecat.transports.smallwebrtc.connection import IceServer
    from pipecat.transports.smallwebrtc.request_handler import (
        ConnectionMode,
        SmallWebRTCPatchRequest,
        SmallWebRTCRequest,
        SmallWebRTCRequestHandler,
    )

@app.server(
    image=image,
    volumes={MODEL_REMOTE: phonon_volume},
    cpu=2.0,
    memory=4096,
    port=PORT,
    startup_timeout=180,
    max_concurrency=1,
    scaledown_window=60,
    exit_grace_period=120,
    compute_region='us-west',
    routing_region='us-west',
    min_containers = 1,
    buffer_containers = 1,
    enable_memory_snapshot=True,
)
@modal.sessioned()
class VoiceBot:
    @modal.enter(snap=True)
    def startup(self) -> None:
        # Volumes are mounted for snapshot capture and again on restore, so
        # this snap=True load can read MODEL_REMOTE. Modal's memory-snapshot
        # guide loads weights from a Volume inside enter(snap=True). Files
        # read here must stay at these paths: volume edits do not refresh the
        # snapshot, and deleting them breaks restore. Snapshotting stays on.

        self.engine = load_engine(VOICE)
        print(f"loaded {self.engine!r}")
        from services.pipecat.vllm_stt import stt_ws_url

        llm_url = llm_base_url()
        stt_url = stt_ws_url()
        print(f"llm {llm_url}")
        print(f"stt {stt_url}")

        ice_servers = []
        for spec in ice_server_specs():
            if isinstance(spec, str):
                ice_servers.append(IceServer(urls=spec))
            else:
                ice_servers.append(IceServer(**spec))
        handler = SmallWebRTCRequestHandler(
            ice_servers=ice_servers or None,
            connection_mode=ConnectionMode.SINGLE,
        )
        self._webrtc = handler
        engine = self.engine

        self._api = FastAPI()
        api = self._api

        @api.get("/health")
        def health(request: Request) -> dict:
            return {
                "status": "ok",
                "transport": "small-webrtc",
                "session_id": request.headers.get("x-modal-server-session-id"),
                "device": engine.device,
                "quant": engine.quant,
                "sample_rate": engine.sample_rate,
                "voices": list(engine.voices),
                "voice": VOICE,
                "stt_base_url": stt_url,
                "llm_base_url": llm_url,
                "ice_servers": ice_server_specs(),
            }

        @api.post("/api/offer")
        async def offer(
            request: Request,
            body: SmallWebRTCRequest,
            background_tasks: BackgroundTasks,
        ):
            session_id = request.headers.get("x-modal-server-session-id", "")

            async def on_connection(connection) -> None:
                background_tasks.add_task(
                    run_bot, connection, engine, session_id, llm_url, stt_url
                )

            return await handler.handle_web_request(body, on_connection)

        @api.patch("/api/offer")
        async def ice_candidates(body: SmallWebRTCPatchRequest) -> dict:
            await handler.handle_patch_request(body)
            return {"status": "success"}

        config = uvicorn.Config(self._api, host="0.0.0.0", port=PORT, log_level="info")
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    @modal.exit()
    def shutdown(self) -> None:
        server = getattr(self, "_server", None)
        thread = getattr(self, "_thread", None)
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)


# Recorded at import time. These steps do not read the phonon checkpoint or
# import the client host. The host module is mounted at /root so a request
# can `import client_host`; VoiceBot's image does not include it.
client_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("fastapi", "httpx")
    .add_local_file(
        Path(__file__).parent / "web" / "client_host.py",
        "/root/client_host.py",
    )
    .add_local_dir(Path(__file__).parent / "client", remote_path="/client")
)


@app.function(
    image=client_image,
    timeout=240,
    scaledown_window=120,
    region='us-west',
    routing_region='us-west',
    min_containers = 1,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def site():
    """Serve the browser client and proxy its WebRTC signaling."""
    from client_host import _client_api

    return _client_api()
