# modal-phonon-voice-bot

A Pipecat voice agent on Modal with Gradium Phonon text-to-speech running in the same CPU container as the agent. Each call is a Modal Session pinned to that container, so Small WebRTC signaling and Phonon share one process. The LLM is a Dedicated Endpoint. Speech-to-text is a separate vLLM server.

```bash
git clone https://github.com/modal-projects/modal-phonon-voice-bot
cd modal-phonon-voice-bot
```

## Phonon in the bot container

Phonon runs in-process on CPU. This demo uses Gradium’s on-device q8 build (`ptts`, `device="cpu"`, `quant="q8"`). The weights are a Modal volume mounted at `/models/phonon` and loaded when the container starts. Synthesis stays in that process, so a remote TTS request is not on the path to the first audio byte. The console Metrics tab shows time to first byte for the call you are on.

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

### 3. Upload the Phonon weights

The checkpoint is gitignored (`phonon-*/`, `*.gguf`) and is not copied into the image. `services/phonon_tts.py` references volume `phonon-7e71a02d` with `create_if_missing` left unset, so the volume has to exist before deploy. The bot mounts it at `/models/phonon`. Checkpoint files belong at the volume root: `config.json`, `model.q8.gguf`, `tokenizer.json`, and `voices/`.

A local copy in this checkout is the gitignored directory `phonon-7e71a02d.200` (`voices/` includes `Freya.safetensors`). From the repo root, create the volume once, then upload. A remote path that ends in `/` is a directory, and the CLI appends the local file or directory name under it. `modal volume put … phonon-7e71a02d.200 /` would store the checkpoint at `/phonon-7e71a02d.200/`, which the mount would not see. These paths put the files at the volume root:

```bash
uv run modal volume create phonon-7e71a02d

uv run modal volume put phonon-7e71a02d phonon-7e71a02d.200/config.json /config.json
uv run modal volume put phonon-7e71a02d phonon-7e71a02d.200/model.q8.gguf /model.q8.gguf
uv run modal volume put phonon-7e71a02d phonon-7e71a02d.200/tokenizer.json /tokenizer.json
uv run modal volume put phonon-7e71a02d phonon-7e71a02d.200/voices /voices
```

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

Open the `site` URL printed by deploy. Connect in the console and allow the microphone. The default voice is Freya (`PHONON_VOICE`). The Metrics tab shows time to first byte for TTS.

ICE is STUN only (`stun:stun.l.google.com:19302`). No TURN server is configured. On a network that cannot use that STUN server, the call may not connect.
