"""
listener.py – MicroPython listener for the cat doorbell.

Hardware: Raspberry Pi Pico 2W + INMP441 I²S MEMS microphone

Wiring (INMP441 → Pico)
-----------------------
    VDD  → 3V3
    GND  → GND
    L/R  → GND          (selects the left I²S slot)
    SCK  → GPIO 10      (BCLK)
    WS   → GPIO 11      (LRCLK / WS — must be SCK + 1)
    SD   → GPIO 12      (data out)

Strategy: Energy-based Voice Activity Detection (VAD)
-----------------------------------------------------
Identical to the desktop version: keep a pre-roll buffer, start recording when
a chunk crosses the RMS threshold, stop after a window of silence, then POST
the captured clip as 16-bit PCM WAV to the YAMNet classifier endpoint.

Pico-specific changes
---------------------
- machine.I2S replaces sounddevice. The INMP441 emits 24-bit data MSB-first
  inside a 32-bit frame; we keep bytes 2–3 of every word, which is the upper
  16 bits of the valid 24-bit signal — already a signed int16.
- The WAV header is built by hand (no scipy on MicroPython).
- urequests sends the multipart body (replicates `requests.post(files=…)`
  on the server side).
- HTTP POST runs on the second core via _thread so audio capture never blocks.
- An in-flight lock drops new clips while a previous POST is still running —
  bounded memory at the cost of skipping back-to-back detections.

Save as `main.py` on the Pico if you want it to auto-start at boot.

Requirements
------------
    MicroPython firmware for RP2350 (Pico 2W) with network + _thread.
"""

import gc
import network
import struct
import time
import _thread
from machine import I2S, Pin

try:
    import urequests as requests
except ImportError:
    import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIFI_SSID = "YOUR_SSID"
WIFI_PASSWORD = "YOUR_PASSWORD"

SERVER_IP = "192.168.1.10"
SERVER_PORT = 9000

# I²S pins. On RP2350 any GPIOs work, but SCK and WS must be consecutive.
SCK_PIN = 10
WS_PIN = 11
SD_PIN = 12

# ---------------------------------------------------------------------------
# Audio configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000          # YAMNet's native sample rate
CHANNELS: int = 1                  # mono — INMP441 with L/R grounded
BITS_OUT: int = 16                 # PCM bit depth sent to the server

CHUNK_MS: int = 100
CHUNK_SAMPLES: int = SAMPLE_RATE * CHUNK_MS // 1000   # 1600
RAW_CHUNK_BYTES: int = CHUNK_SAMPLES * 4              # I²S delivers 32-bit frames
PCM_CHUNK_BYTES: int = CHUNK_SAMPLES * 2

# ---------------------------------------------------------------------------
# VAD / recording parameters (tune to your environment)
# ---------------------------------------------------------------------------
# Threshold is in raw int16 units. The desktop version used 0.02 of full-scale
# float, which corresponds to ~655 here.
RMS_THRESHOLD: int = 600
PRE_ROLL_SEC: float = 0.3
POST_ROLL_SEC: float = 0.8
MIN_SOUND_SEC: float = 0.3
# Capped at 3 s on the Pico — recording + WAV + POST buffer must fit in
# ~520 KB of SRAM. Bump only after measuring headroom.
MAX_SOUND_SEC: float = 3.0

PRE_ROLL_CHUNKS: int = int(PRE_ROLL_SEC * 1000 / CHUNK_MS)
POST_ROLL_CHUNKS: int = int(POST_ROLL_SEC * 1000 / CHUNK_MS)
MIN_SOUND_CHUNKS: int = int(MIN_SOUND_SEC * 1000 / CHUNK_MS)
MAX_SOUND_CHUNKS: int = int(MAX_SOUND_SEC * 1000 / CHUNK_MS)


# ---------------------------------------------------------------------------
# WiFi
# ---------------------------------------------------------------------------
def _connect_wifi() -> None:
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    try:
        wlan.config(pm=0xa11140)  # disable WiFi power-save for low POST latency
    except Exception:
        pass
    if not wlan.isconnected():
        print("Connecting to", WIFI_SSID, "…")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        deadline = time.ticks_add(time.ticks_ms(), 30_000)
        while not wlan.isconnected():
            if time.ticks_diff(deadline, time.ticks_ms()) <= 0:
                raise RuntimeError("WiFi connect timed out")
            time.sleep_ms(200)
    print("WiFi connected:", wlan.ifconfig()[0])


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _convert_32_to_16(raw_buf: bytearray, raw_len: int) -> bytearray:
    """Keep bytes 2–3 of each 32-bit I²S word — i.e. the upper 16 bits of the
    INMP441's valid 24-bit signal, already in little-endian int16 layout."""
    n = raw_len // 4
    out = bytearray(n * 2)
    for i in range(n):
        out[i * 2] = raw_buf[i * 4 + 2]
        out[i * 2 + 1] = raw_buf[i * 4 + 3]
    return out


def _rms_int16(pcm_buf: bytearray) -> int:
    """RMS over every 4th sample — fast enough in pure MicroPython, still
    well above the meow fundamental at 4 kHz effective rate."""
    n = len(pcm_buf) // 2
    if n == 0:
        return 0
    total = 0
    count = 0
    for i in range(0, n, 4):
        b0 = pcm_buf[i * 2]
        b1 = pcm_buf[i * 2 + 1]
        s = b0 | (b1 << 8)
        if s & 0x8000:
            s -= 0x10000
        total += s * s
        count += 1
    return int((total / count) ** 0.5)


def _wav_header(num_samples: int) -> bytes:
    byte_rate = SAMPLE_RATE * CHANNELS * BITS_OUT // 8
    block_align = CHANNELS * BITS_OUT // 8
    data_size = num_samples * CHANNELS * BITS_OUT // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, CHANNELS, SAMPLE_RATE,
        byte_rate, block_align, BITS_OUT,
        b"data", data_size,
    )


# ---------------------------------------------------------------------------
# Multipart upload (matches `requests.post(files=…)` on the server)
# ---------------------------------------------------------------------------
_BOUNDARY = "----PicoCatBoundary7Q9z"


def _build_multipart(pcm_chunks: list) -> bytes:
    """Concatenate WAV header + PCM into a multipart/form-data body."""
    total_pcm = sum(len(c) for c in pcm_chunks)
    num_samples = total_pcm // (BITS_OUT // 8)

    head = (
        "--" + _BOUNDARY + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    tail = ("\r\n--" + _BOUNDARY + "--\r\n").encode()

    parts = [head, _wav_header(num_samples)]
    parts.extend(pcm_chunks)
    parts.append(tail)
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Background HTTP POST (runs on the second core)
# ---------------------------------------------------------------------------
_post_lock = _thread.allocate_lock()


def _post_async(body: bytes) -> None:
    if not _post_lock.acquire(0):
        print("  → previous POST still in flight, dropping clip")
        return

    def _worker(payload):
        try:
            r = requests.post(
                "http://" + SERVER_IP + ":" + str(SERVER_PORT) + "/waveform",
                data=payload,
                headers={
                    "Content-Type":
                        "multipart/form-data; boundary=" + _BOUNDARY,
                },
            )
            print("  → server:", r.text)
            r.close()
        except Exception as e:
            print("  → POST failed:", e)
        finally:
            _post_lock.release()
            gc.collect()

    _thread.start_new_thread(_worker, (body,))


# ---------------------------------------------------------------------------
# Main listener loop
# ---------------------------------------------------------------------------
def listen() -> None:
    """Connect WiFi, open I²S, run VAD, POST detected clips. Blocks until
    KeyboardInterrupt."""
    _connect_wifi()

    audio_in = I2S(
        0,
        sck=Pin(SCK_PIN),
        ws=Pin(WS_PIN),
        sd=Pin(SD_PIN),
        mode=I2S.RX,
        bits=32,
        format=I2S.MONO,
        rate=SAMPLE_RATE,
        ibuf=RAW_CHUNK_BYTES * 4,   # ~400 ms internal DMA buffering
    )

    raw = bytearray(RAW_CHUNK_BYTES)
    pre_roll: list = []
    recording: list = []
    silence_chunks = 0
    active = False

    print("Listening on INMP441 at", SAMPLE_RATE, "Hz")
    print("RMS threshold:", RMS_THRESHOLD, " |  Ctrl-C to quit\n")

    try:
        while True:
            n_bytes = audio_in.readinto(raw)
            if not n_bytes:
                continue

            pcm = _convert_32_to_16(raw, n_bytes)
            loud = _rms_int16(pcm) > RMS_THRESHOLD

            if not active:
                # ── Waiting for a sound to begin ──────────────────────────
                pre_roll.append(pcm)
                if len(pre_roll) > PRE_ROLL_CHUNKS:
                    pre_roll.pop(0)
                if loud:
                    active = True
                    silence_chunks = 0
                    recording = list(pre_roll)
                    pre_roll = []
                    print("Sound detected — recording …")
            else:
                # ── Recording in progress ─────────────────────────────────
                recording.append(pcm)

                if loud:
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                hit_max = len(recording) >= MAX_SOUND_CHUNKS
                went_silent = silence_chunks >= POST_ROLL_CHUNKS

                if hit_max or went_silent:
                    active = False
                    reason = "max duration reached" if hit_max else "silence detected"
                    print("Recording stopped (" + reason + ")")

                    if len(recording) >= MIN_SOUND_CHUNKS:
                        body = _build_multipart(recording)
                        _post_async(body)
                    else:
                        print("  → clip too short, ignoring")

                    recording = []
                    silence_chunks = 0
                    gc.collect()

    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        audio_in.deinit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    listen()
