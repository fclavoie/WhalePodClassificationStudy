# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visual analysis of audio file
#
# This notebook is used to visualize the dB and dB^2 levels as spectrograms of audio files.

# %%
from pathlib import Path
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np


# %%
def generate_spectrograms(file_path, sample_rate=10000, n_fft=512, hop_length=128, n_mels=64):
    """ Generate mel spectrogram and squared mel spectrogram from audio file

    :param file_path: path to audio file
    :param sample_rate: sample rate of audio file
    :param n_fft: number of samples in each window
    :param hop_length: number of samples between windows
    :param n_mels: number of mel bins
    :return: mel spectrogram and squared mel spectrogram
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)

    mel_spec_db = T.AmplitudeToDB()(mel_spec)

    mel_spec_db_squared = np.clip(mel_spec_db.numpy()**2 * 0.1, 0, 100)

    return mel_spec_db.numpy(), mel_spec_db_squared



# %% [markdown]
# ## Selection of audio files

# %%
dataset_synth = Path("dataset_synth")
dataset = Path("dataset")

files_to_process = {
    "Mn_0279.wav": dataset / "Mn_0279.wav",
    "Mn_0058.wav": dataset / "Mn_0058.wav",
    "Mn_0126.wav": dataset / "Mn_0126.wav",
    "Background_0846.wav": dataset / "Background_0846.wav",
    "1877.wav": dataset_synth / "1877.wav",
}


# %% [markdown]
# ## Generation of spectrograms and visualisation

# %%
fig, axes = plt.subplots(2, len(files_to_process), figsize=(15, 6))

for idx, (name, path) in enumerate(files_to_process.items()):
    mel_spec_db, mel_spec_db_squared = generate_spectrograms(path)

    ax_db = axes[0, idx]
    img_db = ax_db.imshow(mel_spec_db[0], aspect="auto", origin="lower", cmap="viridis")
    ax_db.set_title(name)
    ax_db.set_xlabel("Time")
    ax_db.set_ylabel("Frequency")
    fig.colorbar(img_db, ax=ax_db, orientation="vertical")

    ax_db_squared = axes[1, idx]
    img_db_squared = ax_db_squared.imshow(
        mel_spec_db_squared[0], aspect="auto", origin="lower", cmap="plasma"
    )
    ax_db_squared.set_xlabel("Time")
    ax_db_squared.set_ylabel("Frequency")
    fig.colorbar(img_db_squared, ax=ax_db_squared, orientation="vertical")

fig.suptitle("Spectrograms of Synthesized Data - 4s, dB (top) and dB² (bottom)")
plt.tight_layout()
plt.show()
