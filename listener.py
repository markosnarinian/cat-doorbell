"""
listener.py – Microphone listener for the cat doorbell.

VAD strategy: Auto-calibrating energy (RMS) detector
-----------------------------------------------------
silero-vad was evaluated and ruled out: it is trained on human speech only and
produces probabilities below 0.05 on cat meows at any threshold, so it would
never trigger.  webrtcvad is also speech-optimised and has the same limitation.

Energy-based VAD is the correct approach here because we just need to detect
"something audible is happening" – the heavy lifting (cat vs. not-cat) is
handled downstream by YAMNet via backend.py.

What makes this implementation production-quality is the *automatic
noise-floor calibration* at startup: it listens for CALIBRATION_SEC seconds,
measures the ambient RMS, and sets the trigger threshold to
  ambient_rms × THRESHOLD_FACTOR
This adapts to different microphones, rooms, and ambient noise levels without
manual tuning.

State machine
-------------
IDLE  ──(RMS > threshold)──▶  RECORDING ──(silence ≥ POST_ROLL  or  max len)──▶  CLASSIFYING
  ▲                                                                                      │
  └──────────────────────────────────────────────────────────────────────────────────────┘

Requirements
------------
    pip install sounddevice numpy   # already present in the venv
"""

import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from backend import is_cat_present

# ---------------------------------------------------------------------------
# Audio constants (fixed by YAMNet requirements)
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000  # Hz – YAMNet and silero-vad native rate
CHANNELS: int = 1  # mono
CHUNK_DURATION: float = 0.032  # 32 ms chunks  (512 samples)
CHUNK_SAMPLES: int = int(SAMPLE_RATE * CHUNK_DURATION)  # 512

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CALIBRATION_SEC: float = 2.0  # seconds of silence to measure noise floor
THRESHOLD_FACTOR: float = 4.0  # trigger at N × ambient RMS
THRESHOLD_FLOOR: float = 0.005  # minimum threshold regardless of noise floor

# ---------------------------------------------------------------------------
# Recording parameters
# ---------------------------------------------------------------------------
PRE_ROLL_SEC: float = 0.3  # seconds to buffer before the sound starts
POST_ROLL_SEC: float = 0.8  # seconds of sustained silence that end a clip
MIN_CLIP_SEC: float = 0.2  # discard clips shorter than this
MAX_CLIP_SEC: float = 6.0  # hard ceiling – force-stop long recordings

PRE_ROLL_CHUNKS: int = int(PRE_ROLL_SEC / CHUNK_DURATION)
POST_ROLL_CHUNKS: int = int(POST_ROLL_SEC / CHUNK_DURATION)
MIN_CLIP_CHUNKS: int = int(MIN_CLIP_SEC / CHUNK_DURATION)
MAX_CLIP_CHUNKS: int = int(MAX_CLIP_SEC / CHUNK_DURATION)


# ---------------------------------------------------------------------------
# Doorbell
# ---------------------------------------------------------------------------


def ring_doorbell() -> None:
    """
    Called whenever a cat is confirmed at the door.

    Replace / extend with your actual doorbell action, e.g.:
      • trigger a GPIO pin on a Raspberry Pi
      • send a push notification via ntfy / Pushover / Telegram
      • play a chime with sounddevice or playsound
      • publish an MQTT message
    """
    print("🔔  DING DONG!  Cat detected at the door!")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk**2)))


def _calibrate() -> float:
    """
    Record CALIBRATION_SEC of ambient audio and return the measured RMS.
    Blocks until calibration is done.
    """
    n_chunks = int(CALIBRATION_SEC / CHUNK_DURATION)
    collected: list[np.ndarray] = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SAMPLES,
        dtype="float32",
    ) as stream:
        print(
            f"Calibrating noise floor ({CALIBRATION_SEC:.0f}s) – please stay quiet …",
            end="",
            flush=True,
        )
        for _ in range(n_chunks):
            chunk, _ = stream.read(CHUNK_SAMPLES)
            collected.append(chunk[:, 0])

    ambient = _rms(np.concatenate(collected))
    threshold = max(ambient * THRESHOLD_FACTOR, THRESHOLD_FLOOR)
    print(f"\r  ambient RMS = {ambient:.5f}  →  trigger threshold = {threshold:.5f}   ")
    return threshold


def _classify_and_ring(waveform: np.ndarray) -> None:
    """
    Runs in a background daemon thread so audio capture is never stalled.
    """
    duration = len(waveform) / SAMPLE_RATE
    print(f"  → classifying {duration:.2f}s clip …")
    if is_cat_present(waveform):
        ring_doorbell()
    else:
        print("  → no cat detected")


def _submit(recording: list[np.ndarray]) -> None:
    """Validate minimum clip length, then spawn a classification thread."""
    if len(recording) < MIN_CLIP_CHUNKS:
        print("  → clip too short, ignoring")
        return
    waveform = np.concatenate(recording)
    threading.Thread(
        target=_classify_and_ring,
        args=(waveform,),
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Main listener loop
# ---------------------------------------------------------------------------


def listen() -> None:
    """
    Start the microphone listener.  Blocks until Ctrl-C is pressed.

    The sounddevice callback only enqueues raw 32 ms chunks.  All VAD
    state logic runs in the main thread consuming from that queue, which
    keeps the real-time callback minimal and avoids shared-state races.
    """
    threshold = _calibrate()

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"  [sounddevice] {status}", flush=True)
        audio_queue.put(indata[:, 0].copy())

    print(f"Listening at {SAMPLE_RATE} Hz  |  Ctrl-C to quit\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SAMPLES,
        dtype="float32",
        callback=_callback,
    ):
        pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_CHUNKS)
        recording: list[np.ndarray] = []
        silence_chunks: int = 0
        active: bool = False

        try:
            while True:
                chunk = audio_queue.get()
                loud = _rms(chunk) > threshold

                if not active:
                    # ── IDLE: maintain pre-roll buffer ────────────────────
                    pre_roll.append(chunk)
                    if loud:
                        active = True
                        silence_chunks = 0
                        recording = list(pre_roll)  # seed with buffered pre-roll
                        print("Sound detected — recording …")
                else:
                    # ── RECORDING ─────────────────────────────────────────
                    recording.append(chunk)

                    if loud:
                        silence_chunks = 0
                    else:
                        silence_chunks += 1

                    too_long = len(recording) >= MAX_CLIP_CHUNKS
                    went_silent = silence_chunks >= POST_ROLL_CHUNKS

                    if too_long or went_silent:
                        reason = "max duration" if too_long else "silence"
                        print(
                            f"Recording stopped ({reason}), "
                            f"{len(recording) * CHUNK_DURATION:.2f}s captured"
                        )
                        active = False
                        _submit(recording)
                        recording = []
                        silence_chunks = 0

        except KeyboardInterrupt:
            print("\nStopped listening.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    listen()
