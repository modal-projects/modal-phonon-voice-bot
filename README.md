# modal-phonon-voice-bot

A [Pipecat](https://docs.pipecat.ai/) voice agent on Modal with [Gradium Phonon](https://huggingface.co/gradium/phonon-31m-en) text-to-speech running in the same CPU container as the agent. Each call is a [Modal Session](https://modal.com/docs/guide/sticky-sessions) pinned to that container, which makes autoscaling with WebRTC easy on Modal. The LLM is a Modal Dedicated Endpoint using PhoneLLM Mini Alpha 2. Speech-to-text uses Qwen3-ASR served with vLLM - also running on Modal.

https://github.com/user-attachments/assets/85a1f003-c83c-4082-ae1f-f800c96f470e

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

The project name is `modal-phonon-voice-bot`.

### 3. Phonon weights

The bot downloads the q8 English weights from [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) when they are not already cached to the Modal Volume. 
`ptts` reads these files at the volume root: `config.json`, `model.q8.gguf`, `tokenizer.json`, and `default-voice.safetensors` (the voice named `default`).

### 4. Qwen3-ASR

From the repo root,:

```bash
uv run modal deploy services/qwen3-asr-vllm.py
```

### 5. PhoneLLM Endpoint

Deploy [PhoneLLM Mini Alpha 2](https://modal.com/library/pipecat/phonellm-mini-alpha-2) in the same Modal environment as the bot with the name `phonellm-mini-alpha-2`. The bot looks up the Endpoint by name at startup.

### 6. Build the frontend

From `web/`:

```bash
npm install
npm run build
```

`npm run build` runs Vite and writes `../client` which will be attached to frontend image.


### 7. Deploy Bot to Modal

From the repo root, after the frontend build:

```bash
uv run modal deploy phonon_bot.py
```

### 7. Try it out

Open the `site` URL printed by deploy. Click Connect to start a conversations. The Metrics tab shows TTFX metrics for each of the three services: STT, LLM, TTS.

#### Note 
Currently ICE is STUN only. On a network that cannot use that STUN server, e.g. has a restrictive NAT, the call may not connect. See the Pipecat docs to configure a TURN server.
