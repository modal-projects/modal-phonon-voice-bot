"""Phonon checkpoint wiring for the voice bot.

The q8 English files from ``gradium/phonon-31m-en`` live on the
``phonon-7e71a02d`` Volume in the active environment, mounted at
``/models/phonon``.
``ensure_checkpoint`` downloads them on first start and commits the volume.
``load_engine`` builds ``ptts.TTS`` from that mount. ``ptts`` and
``huggingface_hub`` are imported inside those functions so this module stays
light at import time.
"""

import os
import shutil
import tempfile
from pathlib import Path

import modal

MODEL_REMOTE = "/models/phonon"
PHONON_VOLUME = "phonon-7e71a02d"
HF_REPO = "gradium/phonon-31m-en"
# Hub commit whose file listing and sizes were checked. ptts 0.4.0rc3 reads
# these four names from a local directory: config, q8 GGUF, tokenizer JSON,
# and default-voice.safetensors as the voice "default". tokenizer.model,
# model.safetensors, and model.q4k.gguf are published but unused at quant q8.
HF_REVISION = "52433c46975bfa13144c56204c9f4ee1c5b148aa"
CHECKPOINT_FILES: tuple[tuple[str, int], ...] = (
    ("config.json", 2851),
    ("model.q8.gguf", 64512352),
    ("tokenizer.json", 277967),
    ("default-voice.safetensors", 256136),
)
# ptts registers default-voice.safetensors under this name. The repo has no
# Freya, Harper, Sterling, or Toby embedding.
DEFAULT_VOICE = "default"
# No environment_name: lookup and create use the active environment.
# create_if_missing is safe there; it does not target another environment.
phonon_volume = modal.Volume.from_name(PHONON_VOLUME, create_if_missing=True)


def checkpoint_cached(root: str | Path | None = None) -> bool:
    """True when every cached file is present at the published byte size."""
    path = Path(root or MODEL_REMOTE)
    return all(
        (path / name).is_file() and (path / name).stat().st_size == size
        for name, size in CHECKPOINT_FILES
    )


def ensure_checkpoint() -> None:
    """Download the q8 checkpoint onto the mount when it is not already cached.

    Call this from the container that mounts ``phonon_volume``, before
    ``load_engine``. A public repo needs no token.
    """
    if checkpoint_cached():
        return
    from huggingface_hub import snapshot_download

    root = Path(MODEL_REMOTE)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=HF_REPO,
            revision=HF_REVISION,
            local_dir=tmp,
            allow_patterns=[name for name, _size in CHECKPOINT_FILES],
            token=False,
        )
        for name, size in CHECKPOINT_FILES:
            src = Path(tmp) / name
            if not src.is_file() or src.stat().st_size != size:
                raise FileNotFoundError(
                    f"{HF_REPO}@{HF_REVISION} did not provide {name} ({size} bytes)"
                )
            shutil.copyfile(src, root / name)
    phonon_volume.commit()


def load_engine(voice: str | None = None):
    """Construct ``ptts.TTS`` for the checkpoint mounted at ``MODEL_REMOTE``."""
    import ptts

    if voice is None:
        voice = os.environ.get("PHONON_VOICE", DEFAULT_VOICE)
    return ptts.TTS(
        MODEL_REMOTE,
        device="cpu",
        quant="q8",
        voice=voice,
        lang="en",
    )
