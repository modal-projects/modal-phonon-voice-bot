# modal-phonon-voice-bot

A [Pipecat](https://docs.pipecat.ai/) voice agent on Modal with [Gradium Phonon](https://huggingface.co/gradium/phonon-31m-en) text-to-speech running in the same CPU container as the agent. Each call is a [Modal Session](https://modal.com/docs/guide/sticky-sessions) pinned to that container, which makes autoscaling with WebRTC easy on Modal. The LLM is a Modal Dedicated Endpoint using PhoneLLM Mini Alpha 2. Speech-to-text uses Qwen3-ASR served with vLLM - also running on Modal.

<video src="https://modal-cdn.com/cdnbot/phonon_demo_01i9sp3dzg_003b456e.mp4" controls width="100%"></video>

```bash
git clone https://github.com/modal-projects/modal-phonon-voice-bot
cd modal-phonon-voice-bot
```

## Prerequisites

- A Modal account.
- [uv](https://docs.astral.sh/uv/).
- Node.js and npm, to build the console.

## Setup

### 1. Create a Modal account and set up the CLI

Create a Modal account at [modal.com](https://modal.com) if you do not have one.

From the repo root:

```bash
uv sync
uv run modal setup
```

`uv run modal setup` authenticates the CLI.

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
