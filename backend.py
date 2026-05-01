import csv
import io
import os

import kagglehub
import numpy as np
import requests
import scipy
import tensorflow as tf
from dotenv import dotenv_values
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from scipy.io import wavfile

# Download latest version of the model (uses cache if already downloaded)
path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")

# Load the model directly from the local SavedModel directory
model = tf.saved_model.load(path)

# The class map CSV ships alongside the SavedModel in the assets/ folder
class_map_path = os.path.join(path, "assets", "yamnet_class_map.csv")


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names


class_names = class_names_from_csv(class_map_path)


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def infer(waveform):
    scores, _, _ = model(waveform)

    mean_scores = np.mean(scores.numpy(), axis=0)
    sorted_indices = np.argsort(mean_scores)[::-1]

    return [(class_names[i], float(mean_scores[i])) for i in sorted_indices]


def is_cat_present(waveform):
    relevant_classes = ["Cat", "Meow"]
    max_rank = 5
    results = infer(waveform)
    for result in results[0:max_rank]:
        if result[1] > 0.1 and result[0] in relevant_classes:
            return True
    return False


config = dotenv_values(".env")

app = FastAPI()


def ring_doorbell():
    requests.post(
        f"http://{config['IP_ADDRESS']}/rpc",
        json={"id": 1, "method": "Switch.Set", "params": {"id": 0, "on": True}},
    )


@app.post("/waveform")
async def classify(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a WAV file upload, run the YAMNet cat classifier,
    and ring the doorbell if a cat meow is detected.

    Returns JSON with keys:
      - cat_detected (bool)
      - top_results  (list of [class_name, score] pairs)
    """
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    contents = await file.read()
    try:
        sample_rate, wav_data = wavfile.read(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not parse WAV file: {exc}"
        ) from exc

    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    waveform = wav_data / tf.int16.max

    cat_detected = is_cat_present(waveform)

    if cat_detected:
        print("classify: cat detected – ringing doorbell")
        ring_doorbell()
    else:
        print("classify: no cat detected")

    return JSONResponse(
        {
            "cat_detected": cat_detected,
        }
    )


if __name__ == "__main__":
    sample_rate, wav_data = wavfile.read("miaow_16k.wav", "rb")
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    waveform = wav_data / tf.int16.max

    print(is_cat_present(waveform))
