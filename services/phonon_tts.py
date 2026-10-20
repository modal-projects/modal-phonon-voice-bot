"""Phonon checkpoint wiring for the voice bot.

The checkpoint lives on the ``phonon-7e71a02d`` Volume in ``shababo-dev``,
mounted at ``/models/phonon``. ``load_engine`` builds ``ptts.TTS`` from that
mount. ``ptts`` is imported inside the loader so this module stays light at
import time.
"""

import os

import modal

MODEL_REMOTE = "/models/phonon"
PHONON_VOLUME = "phonon-7e71a02d"
MODAL_ENVIRONMENT = "shababo-dev"
# Checkpoint objects sit at the volume root. Mounting the volume at
# MODEL_REMOTE makes them visible to ptts.TTS.
phonon_volume = modal.Volume.from_name(
    PHONON_VOLUME,
    environment_name=MODAL_ENVIRONMENT,
)


def load_engine(voice: str | None = None):
    """Construct ``ptts.TTS`` for the checkpoint mounted at ``MODEL_REMOTE``."""
    import ptts

    if voice is None:
        voice = os.environ.get("PHONON_VOICE", "Freya")
    return ptts.TTS(
        MODEL_REMOTE,
        device="cpu",
        quant="q8",
        voice=voice,
        lang="en",
    )
