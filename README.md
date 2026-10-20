# modal-phonon-voice-bot

A [Pipecat](https://docs.pipecat.ai/) voice agent on Modal with [Gradium Phonon](https://huggingface.co/gradium/phonon-31m-en) text-to-speech running in the same CPU container as the agent. Each call is a [Modal Session](https://modal.com/docs/guide/sticky-sessions) pinned to that container, which makes autoscaling with WebRTC easy on Modal. The LLM is a Modal Dedicated Endpoint using PhoneLLM Mini Alpha 2. Speech-to-text uses Qwen3-ASR served with vLLM - also running on Modal.

<video src="https://modal-cdn.com/cdnbot/phonon_demo_01i9sp3dzg_003b456e.mp4" controls width="100%"></video>

```bash
git clone https://github.com/modal-projects/modal-phonon-voice-bot
cd modal-phonon-voice-bot
```

## Phonon in the bot container

Phonon runs in-process on CPU. This demo uses Gradium’s on-device q8 build (`ptts`, `device="cpu"`, `quant="q8"`). The weights are cached on a Modal volume and loaded when the container starts. Synthesis stays in that process, so we save network hops on the path to the first audio byte.

## One session per call

`VoiceBot` is a sessioned server. One call stays on one container. The Small WebRTC peer connection lives in memory in that process, next to Phonon, so the SDP offer and the ICE candidate patches have to reach that same process.

The browser talks only to `site`, the public ASGI app on the same origin. A browser preflight cannot send `Modal-Authorization` to the bot, so `site` stores the session token in an HttpOnly cookie and forwards `/api/offer` (POST for the offer, PATCH for ICE candidates).

WebRTC media is UDP and does not keep the Modal session alive. While the call is connected, the console calls `/api/keepalive` every 30 seconds. `site` forwards that to the pinned container. The session idle timeout is 10 minutes.

## PhoneLLM Mini

Replies come from the PhoneLLM Mini Alpha 2 Dedicated Endpoint, `phonellm-mini-alpha-2`. At startup the bot looks up that endpoint and calls its OpenAI-compatible chat API. The backing app is `ep-phonellm-mini-alpha-2` and the server class is `Server`. This repo does not define the endpoint; deploy [PhoneLLM Mini Alpha 2](https://modal.com/library/pipecat/phonellm-mini-alpha-2) in the same Modal environment as the bot.

In the same bot container, Silero VAD and Smart Turn decide when a user turn ends. Speech-to-text is Qwen3-ASR (`Qwen/Qwen3-ASR-0.6B`) on a realtime websocket to the `qwen3-asr-vllm` app.

## Prerequisites

- A Modal account. Authenticate the CLI from this repo after `uv sync`, with `uv run modal setup` (the same binary is `.venv/bin/modal`).
- Python 3.13+ and [uv](https://docs.astral.sh/uv/).
- Node.js and npm, to build the console.

Commands use the default Modal environment for your profile.

## Setup

### 1. Create a Modal account and set up the CLI

Create a Modal account at [modal.com](https://modal.com) if you do not have one.

From the repo root:

```bash
uv sync
uv run modal setup
```

`uv run modal setup` authenticates the CLI. The same binary is `.venv/bin/modal`. Deploy and volume commands use that profile’s default environment.

### 2. Install Python dependencies

From the repo root:

```bash
uv sync
```

The project name is `modal-phonon-voice-bot`. The local environment is the Modal CLI. Pipecat, `ptts`, and the rest of the bot’s Python packages install into the container image at deploy time.

### 3. Phonon weights

The checkpoint is not copied into the image. `services/phonon_tts.py` references volume `phonon-7e71a02d` with `create_if_missing=True` and no environment name, so Modal creates it in your profile’s default environment if it does not already exist. The bot mounts it at `/models/phonon`.

The first container start downloads the q8 English files from [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) when they are not already on that volume, then commits them. Later containers reuse the cache. The repository is public, so a Hugging Face token is not required.

`ptts` reads these files at the volume root: `config.json`, `model.q8.gguf`, `tokenizer.json`, and `default-voice.safetensors` (the voice named `default`).

### 4. Build the frontend

`client/` is gitignored. The site image copies it at deploy time, and deploy fails if the directory is missing. From `web/`:

```bash
npm install
npm run build
```

`npm run build` runs Vite and writes `../client`.

### 5. Qwen3-ASR and the PhoneLLM endpoint

Both have to be deployed in the same environment as the bot.

Speech-to-text is app `qwen3-asr-vllm`, server class `VLLMServer`. From the repo root, if you still need to deploy it:

```bash
uv run modal deploy services/qwen3-asr-vllm.py
```

This repo does not define the LLM endpoint `phonellm-mini-alpha-2` (backing app `ep-phonellm-mini-alpha-2`, server class `Server`). Deploy [PhoneLLM Mini Alpha 2](https://modal.com/library/pipecat/phonellm-mini-alpha-2) in the same Modal environment as the bot. The bot looks up `phonellm-mini-alpha-2` at startup.

### 6. Deploy

From the repo root, after the frontend build:

```bash
uv run modal deploy phonon_bot.py
```

This publishes app `phonon-pipecat-bot`: the `VoiceBot` server and the `site` function. The build picks up `services/`, `web/client_host.py`, and `client/` from this directory. `site` is in `us-west`.

### 7. Try it out

Open the `site` URL printed by deploy. Connect in the console and allow the microphone. The default voice is `default` (`PHONON_VOICE`), from `default-voice.safetensors`. The Metrics tab shows time to first byte for TTS.

ICE is STUN only (`stun:stun.l.google.com:19302`). No TURN server is configured. On a network that cannot use that STUN server, the call may not connect.
