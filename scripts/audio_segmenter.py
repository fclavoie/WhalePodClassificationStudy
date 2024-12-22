"""This module handles the extraction of audio segments from large .x.flac PIFSC audio files and the generation of spectrograms for these segments.

Configurations:
    FOLDER: The root directory containing the audio files.
    WINDOW_LENGTH: The duration of each extracted segment in seconds.
"""

import random
from pathlib import Path

import ffmpeg as ff
import matplotlib.pyplot as plt
import polars as pl
import soundfile as sf

FOLDER = Path.cwd()
WINDOW_LENGTH = 4
random.seed(42)


def generate_spectrogram(wav_path, output_path):
    """Generates a spectrogram from a WAV file and saves it as an image.

    :param wav_path: The path to the input WAV file.
    :param output_path: The path to save the output spectrogram image.
    """
    data, samplerate = sf.read(wav_path)
    plt.specgram(data, Fs=samplerate, NFFT=1024, noverlap=512, cmap="viridis")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def extract_audio_segments(
    file_path, time_ranges, segment_count, segment_duration, output_dir
):
    """Extracts audio segments from a larger audio file and saves them to the output directory.

    :param file_path: The path to the input audio file.
    :param time_ranges: A list of time ranges to extract segments from.
    :param segment_count: The number of segments to extract.
    :param segment_duration: The duration of each segment in seconds.
    :param output_dir: The directory to save the extracted audio segments.
    """
    file_stem = file_path.name
    output_dir = output_dir / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "annotations.csv"

    if csv_file.exists():
        df = pl.read_csv(csv_file)
    else:
        df = pl.DataFrame(
            {
                "filename": pl.Series([], dtype=pl.Utf8),
                "begin_seek": pl.Series([], dtype=pl.Int64),
                "duration": pl.Series([], dtype=pl.Int64),
                "count": pl.Series([], dtype=pl.Int64),
                "keep": pl.Series([], dtype=pl.Int64),
            }
        )

    annotations = []

    all_segments = []
    for start_time, end_time in time_ranges:
        available_duration = end_time - start_time
        max_segments = available_duration // segment_duration

        if segment_count > max_segments:
            raise ValueError(
                f"Not enough space in range ({start_time}, {end_time}) for {segment_count} segments without overlap."
            )

        segments = random.sample(
            range(start_time, start_time + available_duration - segment_duration),
            min(segment_count, max_segments),
        )
        all_segments.extend(segments)

    all_segments = sorted(all_segments)

    for segment_start in all_segments:
        wav_output = output_dir / f"{segment_start}.wav"
        spectrogram_output = output_dir / f"{segment_start}.png"

        ff.input(file_path, ss=segment_start, t=segment_duration).output(
            str(wav_output), format="wav", audio_bitrate="10k"
        ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)

        generate_spectrogram(wav_output, spectrogram_output)

        annotations.append(
            {
                "filename": wav_output.name,
                "begin_seek": segment_start,
                "duration": segment_duration,
                "count": None,
                "keep": None,
            }
        )

    df = df.vstack(pl.DataFrame(annotations))
    df.write_csv(csv_file)


file_path = (
    FOLDER
    / "pifsc/audio/hawaii/pipan_hawaii_16/audio/Hawaii_K_16_140302_084730.df20.x.flac"
)
print(file_path)


output_dir = FOLDER / "handpicked_samples"
time_ranges = [(27799, 28263)]
segment_count = 20

extract_audio_segments(file_path, time_ranges, segment_count, WINDOW_LENGTH, output_dir)
