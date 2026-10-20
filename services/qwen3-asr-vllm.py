"""Qwen3-ASR realtime transcription on vLLM, served on Modal.

Serves ``ws://<host>/v1/realtime`` for ``Qwen/Qwen3-ASR-0.6B`` using the
``Qwen3ASRRealtimeGeneration`` architecture. That architecture is not selected
from the checkpoint's ``config.json``, so the server passes ``--hf-overrides``.

Audio on the socket is base64 PCM16, 16 kHz, mono. The server emits
``transcription.delta`` while audio arrives and ``transcription.done`` when the
client commits ``final=true``.

vLLM keys that state by the WebSocket. A dropped connection aborts generation,
and the next socket starts empty. Hold the socket open for continuity.

Deploy:
    modal deploy services/qwen3-asr-vllm.py
"""

import logging

import modal

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

VLLM_PORT = 8000
VLLM_MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
# Not auto-detected. Without this override vLLM loads the batch ASR arch and
# /v1/realtime cannot run the streaming buffer.
REALTIME_HF_OVERRIDES = '{"architectures":["Qwen3ASRRealtimeGeneration"]}'
# Latest PyPI release. CUDA 12.9 wheel: the default 0.32.0 wheel is CUDA 13
# and fails on this base image with ImportError: libcudart.so.13.
VLLM_VERSION = "0.32.0"
VLLM_WHEEL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.32.0/"
    "vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
# 0.6B fits a 24GB card. Realtime re-encodes the growing utterance, so size
# MAX_CONNECTIONS from a benchmark before raising it.
GPU_TYPE = "H100"

# One open /v1/realtime socket is one engine sequence and one in-flight
# request. These three stay equal:
#   * `--max-num-seqs` — sequences vLLM batches on the GPU
#   * `target_concurrency` — sockets per container before Modal scales out
#   * `max_concurrency` — hard cap; a full container returns 429
MAX_CONNECTIONS = 4

app = modal.App("qwen3-asr-vllm")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_cache_path = "/root/.cache/huggingface"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_cache_path = "/root/.cache/vllm"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install(
        f"vllm[audio] @ {VLLM_WHEEL}",
        extra_options="--torch-backend=cu129",
        gpu=GPU_TYPE,
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": hf_cache_path,
        "VLLM_CACHE_PATH": vllm_cache_path,
        "TORCH_CPP_LOG_LEVEL": "FATAL",
    })
    .entrypoint([])
)

with vllm_image.imports():
    import aiohttp


@app.server(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={hf_cache_path: hf_cache_vol, vllm_cache_path: vllm_cache_vol},
    port=VLLM_PORT,
    startup_timeout=20 * 60,
    min_containers=1,
    target_concurrency=MAX_CONNECTIONS,
    max_concurrency=MAX_CONNECTIONS,
    unauthenticated=True,
    compute_region="us-west",
    routing_region="us-west",
    enable_memory_snapshot=True,
)
class VLLMServer:
    @modal.enter()
    async def launch_vllm_server(self) -> None:
        import subprocess

        cmd = [
            "vllm",
            "serve",
            VLLM_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-num-seqs",
            str(MAX_CONNECTIONS),
            "--gpu-memory-utilization",
            "0.9",
            "--hf-overrides",
            REALTIME_HF_OVERRIDES,
        ]
        logger.info("Starting vLLM server: %s", cmd)
        self.vllm_proc = subprocess.Popen(cmd)  # noqa: ASYNC220

        await self.wait_for_server()
        version = await self.server_version()
        logger.info(
            "vLLM realtime server ready (version=%s, pinned=%s, model=%s, connections=%d)",
            version,
            VLLM_VERSION,
            VLLM_MODEL_NAME,
            MAX_CONNECTIONS,
        )

    async def wait_for_server(self, healthcheck_timeout: float = 18 * 60) -> None:
        """Block until the vLLM server reports healthy, or raise on timeout/crash."""
        import asyncio
        import time

        start_time = time.time()
        count = 0
        while time.time() - start_time < healthcheck_timeout:
            if self.vllm_proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited during startup (code={self.vllm_proc.returncode})"
                )
            if await self.healthcheck():
                logger.info("vLLM server is ready after %d checks", count)
                return
            count += 1
            if count % 10 == 0:
                logger.warning("vLLM server NOT ready after %d checks", count)
            await asyncio.sleep(1)
        raise TimeoutError("vLLM server failed to start within timeout period")

    async def healthcheck(self) -> bool:
        """Return True when the vLLM server's /health endpoint responds 200."""
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"http://localhost:{VLLM_PORT}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response,
            ):
                return response.status == 200
        except aiohttp.ClientError:
            return False

    async def server_version(self) -> str:
        """Fetch the running vLLM version so the shipped build is traceable."""
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"http://localhost:{VLLM_PORT}/version",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response,
            ):
                if response.status == 200:
                    return (await response.json()).get("version", "unknown")
        except aiohttp.ClientError:
            pass
        return "unknown"

    @modal.exit()
    def stop(self) -> None:
        proc = getattr(self, "vllm_proc", None)
        if proc and proc.poll() is None:
            proc.terminate()


def _wav_to_pcm16_16k(data: bytes) -> bytes:
    """Convert a WAV payload to little-endian PCM16 mono at 16 kHz."""
    import array
    import io
    import wave

    with wave.open(io.BytesIO(data), "rb") as wf:
        channels, sampwidth, rate, nframes = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth != 2:
        raise SystemExit(f"expected 16-bit wav, got {sampwidth * 8}-bit")
    samples = array.array("h")
    samples.frombytes(raw)
    if channels == 2:
        samples = array.array("h", ((samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)))
    elif channels != 1:
        raise SystemExit(f"expected mono or stereo wav, got {channels} channels")
    if rate != 16000:
        ratio = 16000 / rate
        out_len = int(len(samples) * ratio)
        resampled = array.array("h")
        last = len(samples) - 1
        for i in range(out_len):
            src = i / ratio
            j = int(src)
            frac = src - j
            a = samples[min(j, last)]
            b = samples[min(j + 1, last)]
            resampled.append(int(a + (b - a) * frac))
        samples = resampled
    return samples.tobytes()


def _ws_url(http_url: str) -> str:
    base = http_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base.removeprefix("https://") + "/v1/realtime"
    if base.startswith("http://"):
        return "ws://" + base.removeprefix("http://") + "/v1/realtime"
    raise SystemExit(f"unexpected server URL: {http_url}")


if __name__ == "__main__":
    import asyncio
    import base64
    import json

    import aiohttp
    import websockets

    async def main() -> None:
        server = modal.Server.from_name("qwen3-asr-vllm", "VLLMServer")
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
        async with (
            aiohttp.ClientSession() as session,
            session.get(audio_url, timeout=aiohttp.ClientTimeout(total=60)) as response,
        ):
            pcm = _wav_to_pcm16_16k(await response.read())
        async with websockets.connect(_ws_url(server.get_url()), max_size=None) as ws:
            created = json.loads(await ws.recv())
            print("session.created", created)
            await ws.send(json.dumps({"type": "session.update", "model": VLLM_MODEL_NAME}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            chunk = 4096
            for i in range(0, len(pcm), chunk):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm[i : i + chunk]).decode(),
                        }
                    )
                )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            while True:
                event = json.loads(await ws.recv())
                kind = event.get("type")
                if kind == "transcription.delta":
                    print(event.get("delta", ""), end="", flush=True)
                elif kind == "transcription.done":
                    print("\ntranscription.done", event.get("text"))
                    break
                elif kind == "error":
                    raise SystemExit(f"realtime error: {event}")
                else:
                    print("event", event)

    asyncio.run(main())


# =============================================================================
# NOTES
# =============================================================================
# - A reconnect does not resume. vLLM drops the audio buffer and text prefix
#   when the socket closes. The client has to send the audio again.
# - An open socket counts as one concurrent request. target_concurrency and
#   max_concurrency match --max-num-seqs so a full GPU returns 429 and Modal
#   scales out instead of queueing more streams on it.
# - Realtime Qwen3-ASR re-encodes the accumulated utterance on each segment
#   (default about 2s of new audio). Cost grows with utterance length. Size
#   MAX_CONNECTIONS from audio-seconds/sec and time-to-first-delta, not from
#   decode-bound LLM concurrency.
