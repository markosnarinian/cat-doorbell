import csv
import os

import kagglehub
import numpy as np
import scipy
import tensorflow as tf
from IPython.display import Audio
from scipy.io import wavfile

# Download latest version of the model (uses cache if already downloaded)
path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")

# Load the model directly from the local SavedModel directory
model = tf.saved_model.load(path)

# The class map CSV ships alongside the SavedModel in the assets/ folder
class_map_path = os.path.join(path, "assets", "yamnet_class_map.csv")


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])

    return class_names


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


class_names = class_names_from_csv(class_map_path)

# wav_file_name = 'speech_whistling2.wav'
wav_file_name = "miaow_16k.wav"
sample_rate, wav_data = wavfile.read(wav_file_name, "rb")
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Show some basic information about the audio.
duration = len(wav_data) / sample_rate
print(f"Sample rate: {sample_rate} Hz")
print(f"Total duration: {duration:.2f}s")
print(f"Size of the input: {len(wav_data)}")

# Listening to the wav file.
Audio(wav_data, rate=sample_rate)

waveform = wav_data / tf.int16.max

# Run the model, check the output.
scores, embeddings, spectrogram = model(waveform)

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()

# Rank classes by their mean score across all frames
mean_scores = np.mean(scores_np, axis=0)
top_n = 10
top_class_indices = np.argsort(mean_scores)[::-1][:top_n]

print("Top sounds detected:")
for rank, idx in enumerate(top_class_indices, start=1):
    print(f"  {rank:2}. {class_names[idx]:<30s}  (score: {mean_scores[idx]:.4f})")
