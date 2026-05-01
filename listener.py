"""
listener.py – Microphone listener for the cat doorbell.

Strategy: Energy-based Voice Activity Detection (VAD)
------------------------------------------------------
1. Keep a short circular *pre-roll* buffer so the attack of a sound is never lost.
2. When a chunk exceeds the RMS threshold, start recording
   (pre-roll + live audio).
3. Stop recording after POST_ROLL_SEC of continuous silence,
   or when MAX_SOUND_SEC is reached.
4. If the captured clip is long enough, send it to the YAMNet classifier
   running in a background thread so audio capture is never blocked.
5. Ring the doorbell if a cat is detected.

Why not stream continuously?
-----------------------------
YAMNet analyses 0.96 s patches internally.  Feeding it silence wastes CPU/GPU
and risks splitting a meow across two windows.  Capturing the full discrete
sound event and classifying it once is both more accurate and more efficient.

Alternatives
------------
- webrtcvad  : Google's binary frame-level VAD (pip install webrtcvad)
- silero-vad : ML-based VAD, state-of-the-art accuracy
- Overlap-stride streaming: classify a rolling 1-2 s window on every new chunk
  (simpler but always-on CPU cost)

Requirements
------------
    pip install sounddevice numpy
"""

import io
import queue
import threading
import time
from collections import deque

import numpy as np
import requests
import scipy.io.wavfile as wav
import sounddevice as sd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IP_ADDRESS = "192.168.1.10"
PORT = 9000

# ---------------------------------------------------------------------------
# Audio configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000  # Hz  – YAMNet's native sample rate
CHANNELS: int = 1  # mono
CHUNK_DURATION: float = 0.1  # seconds per audio chunk (100 ms)
CHUNK_SAMPLES: int = int(SAMPLE_RATE * CHUNK_DURATION)

# ---------------------------------------------------------------------------
# VAD / recording parameters  (tune these to your environment)
# ---------------------------------------------------------------------------
RMS_THRESHOLD: float = 0.02  # normalised RMS level considered "not silence"
PRE_ROLL_SEC: float = 0.3  # seconds of audio kept before the sound starts
POST_ROLL_SEC: float = 0.8  # seconds of silence that signals end-of-sound
MIN_SOUND_SEC: float = 0.3  # clips shorter than this are discarded
MAX_SOUND_SEC: float = 6.0  # hard ceiling – force-stop the recording

PRE_ROLL_CHUNKS: int = int(PRE_ROLL_SEC / CHUNK_DURATION)
POST_ROLL_CHUNKS: int = int(POST_ROLL_SEC / CHUNK_DURATION)
MIN_SOUND_CHUNKS: int = int(MIN_SOUND_SEC / CHUNK_DURATION)
MAX_SOUND_CHUNKS: int = int(MAX_SOUND_SEC / CHUNK_DURATION)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_loud(chunk: np.ndarray) -> bool:
    """Return True when the chunk's RMS energy exceeds the threshold."""
    rms = float(np.sqrt(np.mean(chunk**2)))
    return rms > RMS_THRESHOLD


def _post_waveform(waveform: np.ndarray) -> dict:
    """
    Encode *waveform* (float32, [-1, 1]) as a 16-bit PCM WAV and POST it to
    the YAMNet classifier endpoint.  Returns the parsed JSON response.
    """
    # Convert float32 → int16 so the backend can divide by int16.max correctly
    pcm = (waveform * np.iinfo(np.int16).max).astype(np.int16)

    buf = io.BytesIO()
    wav.write(buf, SAMPLE_RATE, pcm)
    buf.seek(0)

    response = requests.post(
        f"http://{IP_ADDRESS}:{PORT}/waveform",
        files={"file": ("clip.wav", buf, "audio/wav")},
        timeout=30,
    )
    return response.json()


# ---------------------------------------------------------------------------
# Main listener loop
# ---------------------------------------------------------------------------


def listen() -> None:
    """
    Start the microphone listener.  Blocks until Ctrl-C is pressed.

    The sounddevice callback only enqueues raw audio chunks; all VAD logic
    runs in the main thread consuming from that queue, so there is no
    shared-state concurrency between the callback and the detector.
    """
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"  [sounddevice] {status}", flush=True)
        # Extract mono channel and hand off to the main thread
        audio_queue.put(indata[:, 0].copy())

    print(f"Listening on default microphone at {SAMPLE_RATE} Hz …")
    print(f"RMS threshold: {RMS_THRESHOLD}  |  Ctrl-C to quit\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SAMPLES,
        dtype="float32",  # values already in [-1, 1] – no normalisation needed
        callback=_audio_callback,
    ):
        pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_CHUNKS)
        recording: list[np.ndarray] = []
        silence_chunks: int = 0
        active: bool = False

        try:
            while True:
                chunk = audio_queue.get()

                if not active:
                    # ── Waiting for a sound to begin ──────────────────────
                    pre_roll.append(chunk)
                    if _is_loud(chunk):
                        active = True
                        silence_chunks = 0
                        recording = list(pre_roll)  # seed with buffered pre-roll
                        print("Sound detected — recording …")
                else:
                    # ── Recording in progress ─────────────────────────────
                    recording.append(chunk)

                    if _is_loud(chunk):
                        silence_chunks = 0
                    else:
                        silence_chunks += 1

                    hit_max = len(recording) >= MAX_SOUND_CHUNKS
                    went_silent = silence_chunks >= POST_ROLL_CHUNKS

                    if hit_max or went_silent:
                        active = False
                        reason = (
                            "max duration reached" if hit_max else "silence detected"
                        )
                        print(f"Recording stopped ({reason})")

                        if len(recording) >= MIN_SOUND_CHUNKS:
                            waveform = np.concatenate(recording)
                            threading.Thread(
                                target=_post_waveform,
                                args=(waveform,),
                                daemon=True,
                            ).start()
                        else:
                            print("  → clip too short, ignoring")

                        recording = []
                        silence_chunks = 0

        except KeyboardInterrupt:
            print("\nStopped listening.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    listen()
