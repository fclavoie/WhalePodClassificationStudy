"""
This module handles the sample rate resampling of audio files to a target sample rate and duration.
It reads audio files from a source directory, resamples them, pads or truncates them to a fixed number of samples, and saves them to a destination directory.

Configurations:
    SOURCE_ROOT: The root directory containing the original audio files.
    DEST_ROOT: The root directory where the resampled audio files will be saved.
    AUDIO_EXT: The file extension of the audio files to be processed.
    TARGET_SR: The target sample rate for resampling.
    DURATION_S: The target duration of the audio files in seconds.
    NUM_SAMPLES: The number of samples corresponding to the target duration at the target sample rate.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

SOURCE_ROOT = Path("../dataset_split").resolve()
DEST_ROOT = Path("../dataset_16khz").resolve()
AUDIO_EXT = ".wav"

TARGET_SR = 16000
DURATION_S = 4
NUM_SAMPLES = TARGET_SR * DURATION_S


def resample(wav_in: str, wav_out: str, target_sr: int, sample_count: int):
    """Reads an input WAV file, resamples it to the target sample rate,
    truncates or pads it to the specified number of samples, and saves it to the output path.

    :param wav_in: The path to the input WAV file.
    :param wav_out: The path to the output WAV file.
    :param target_sr: The target sample rate for resampling.
    :param sample_count: The number of samples to pad or truncate to.
    """
    audio, sr = torchaudio.load(wav_in)

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    length = audio.shape[-1]
    if length > sample_count:
        audio = audio[..., :sample_count]
    elif length < sample_count:
        nb_to_pad = sample_count - length
        audio = F.pad(audio, (0, nb_to_pad))

    Path(wav_out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(wav_out, audio, sample_rate=target_sr)


def process_directory(
    source_dir: Path, dest_dir: Path, target_sr: int, sample_count: int
):
    """Processes all audio files in the source directory by resampling them and saving them to the destination directory.

    :param source_dir: The source directory containing the original audio files.
    :param dest_dir: The destination directory to save the resampled audio files.
    :param target_sr: The target sample rate for resampling.
    :param sample_count: The number of samples to pad or truncate to.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in source_dir.glob(f"*{AUDIO_EXT}"):
        dest_wav_path = dest_dir / wav_path.name
        resample(str(wav_path), str(dest_wav_path), target_sr, sample_count)


if __name__ == "__main__":
    for folder in ["train", "test"]:
        source_split = (SOURCE_ROOT / folder).resolve()
        dest_split = (DEST_ROOT / folder).resolve()
        process_directory(source_split, dest_split, TARGET_SR, NUM_SAMPLES)
