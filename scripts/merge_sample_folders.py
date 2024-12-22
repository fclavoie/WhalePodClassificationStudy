"""
This module processes and merges audio samples from multiple source directories into a single destination directory.
It also generates unique filenames for the audio files and compiles annotations into a single CSV file.

Configurations:
    source_dirs: List of source directories containing the audio samples and annotations.
    destination_dir: Directory to save the processed audio samples and compiled annotations.
    output_annotations: Path to the output annotations CSV file.
"""

import hashlib
import shutil
from pathlib import Path

import polars as pl

source_dirs = [
    Path("../handpicked_samples/Hawaii_K_10_110221_194400.df20.x.flac"),
    Path("../handpicked_samples/Hawaii_K_16_140302_084730.df20.x.flac"),
    Path("../handpicked_samples/Hawaii_K_20_150423_064030.df32.x.flac"),
]
destination_dir = Path("../dataset_test")
destination_dir.mkdir(parents=True, exist_ok=True)
output_annotations = destination_dir / "annotations.csv"


def generate_hash(data, length=6):
    """Generates a hash for the given data.

    :param data: The input data to hash.
    :param length: The length of the hash to return, defaults to 6.
    :return: The generated hash.
    """
    return hashlib.md5(data.encode()).hexdigest()[:length]


merged_annotations = pl.DataFrame(
    schema={
        "extraction": pl.Utf8,
        "pifsc_index": pl.Float64,
        "label": pl.Utf8,
        "source": pl.Utf8,
        "begin_seek": pl.Float64,
        "end_seek": pl.Float64,
        "count": pl.Int64,
    }
)

# Process each source directory
for source_dir in source_dirs:
    source_name = source_dir.name

    wav_files = list(source_dir.glob("*.wav"))
    annotations_path = source_dir / "annotations.csv"

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found in {source_dir}")

    annotations = pl.read_csv(annotations_path)

    site_code = "H" + source_name.split("_K_")[1].split("_")[0]
    for wav_file in wav_files:
        # Generate a unique name for the extracted .wav file
        hash = generate_hash(wav_file.stem, length=6)
        new_name = f"Hp_{wav_file.stem}.{site_code}.{hash}.wav"
        destination_path = destination_dir / new_name

        # Copy the .wav file to the destination
        shutil.copy(wav_file, destination_path)

        # Extract fields for annotation
        begin_seek = float(wav_file.stem)
        annotation_row = annotations.filter(pl.col("filename") == wav_file.name)
        if annotation_row.height == 0:
            raise ValueError(f"No matching annotation found for {wav_file.name}")

        duration = float(annotation_row["duration"].item())
        whale_count = int(annotation_row["count"].item())
        end_seek = begin_seek + duration

        # Append to merged annotations
        merged_annotations = merged_annotations.vstack(
            pl.DataFrame(
                {
                    "extraction": [new_name],
                    "pifsc_index": [None],
                    "label": ["Mn"],
                    "source": [source_name],
                    "begin_seek": [begin_seek],
                    "end_seek": [end_seek],
                    "count": [whale_count],
                }
            )
        )

merged_annotations.write_csv(output_annotations)
