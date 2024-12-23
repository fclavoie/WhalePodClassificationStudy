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
# # Segment extraction from .x.flac files

# %%
from pathlib import Path
from datetime import datetime, timezone
import csv
import subprocess
import json
import struct

import polars as pl
import ffmpeg as ff
import numpy as np
import soundfile as sf

root = Path().resolve()


# %% [markdown]
# ## Data loading
#
# First, we read the annotations of the dataset provided by PIFSC (https://doi.org/10.25921/Z787-9Y54).
# We also add columns for ease the extraction process.

# %%
csv_file = "pifsc/pifsc_products_detections_annotations.csv"
df = pl.read_csv(csv_file)
df = df.with_row_index()

df = df.with_columns(
    [
        pl.col("begin_utc").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f%z"),
        pl.col("end_utc").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f%z"),
    ]
)

df = df.with_columns(
    (df["end_rel_subchunk"] - df["begin_rel_subchunk"]).alias("duration")
)

print(df.select("audit_name").unique())
print(df.select(pl.col("audit_name").value_counts()))

# %%
df = df.filter(
    pl.col("flac_compressed_xwav_object").str.contains("/hawaii/")
)

# %% [markdown]
# ## Extracting subchunk metadata
# Using preprocess metadata file (created with metadata_extraction.py), we extract seek locations.

# %%
files_of_interest = df["flac_compressed_xwav_object"].map_elements(lambda x: Path(x).stem, return_dtype=pl.String).unique()

meta_dir = Path("./pifsc/audio/hawaii/meta/")
metadata = {}
for filename in files_of_interest:
    json_file = meta_dir / f"{filename}.json"
    if json_file.exists():
        with open(json_file, "r") as f:
            metadata[filename] = json.load(f)

def calculate_seek(byte_loc, sample_rate, bits_per_sample=16):
    return byte_loc / (sample_rate * (bits_per_sample / 8))

def calculate_read_segment(row):
    filename = Path(row[2]).stem
    subchunk_index = row[3]

    file_meta = metadata.get(filename)
    if file_meta:
        raw_files = file_meta["raw_files"]
        if subchunk_index < len(raw_files):
            raw_file = raw_files[subchunk_index]
            begin_seek = calculate_seek(
                raw_file["byte_loc"], raw_file["sample_rate"]
            )
            end_seek = begin_seek + row[11]
            return begin_seek, end_seek
    return None, None

read_times = df.map_rows(calculate_read_segment)
df = df.hstack(read_times)
df = df.rename({"column_0": "begin_seek", "column_1": "end_seek"})


# %%
print(df[-5]["flac_compressed_xwav_object"].item())
df[-5]


# %%
def correct_path(flac_path):
    """ Correct the path of the flac file to point toward a local folder."""
    path = flac_path.replace(
        "gs://noaa-passive-bioacoustic/pifsc/audio/pipan/", "pifsc/audio/"
    )
    return root / Path(path)


# %%
track = df[10]
track

# %% [markdown]
# ## Segment extraction
#
# We first filter the annotations to retains only usable segments of humpback whales and background noises, then we extract them to individual .wav files.
#

# %%
df_mn = df.filter(
    (pl.col("label") == "Mn")
    # & (pl.col("duration").is_between(1, 2))
    & (pl.col("audit_name").is_in(["initial", "segments", "validation", "model_2"]))
    & (pl.col("begin_seek").is_not_null())
).unique(
    subset=["flac_compressed_xwav_object", "subchunk_index", "label", "begin_seek"]
)
len(df_mn)

# %%
df_f = df.filter((pl.col("audit_name").is_in(["segments"]))).unique(
    "flac_compressed_xwav_object"
)
print(len(df_f))
for file in df_f.get_column("flac_compressed_xwav_object"):
    print(file)

# %%
df_noise = df.filter(
    (pl.col("label") != "Mn") & (pl.col("begin_seek").is_not_null())
).unique(
    subset=["flac_compressed_xwav_object", "subchunk_index", "label", "begin_seek"]
)
len(df_noise)


# %%
def extract(samples, destination, seed=0, registry: csv.writer = None, default_count=0):
    """ Extract samples from the .x.flac files and save them as wav files.
    
    :param samples: A list of samples to extract.
    :param destination: The folder where the samples will be saved.
    :param seed: The seed to use for the naming of the files.
    :param registry: A csv writer to save the information about the extracted samples.
    :param default_count: The default count of whale vocalisation to be used. 1 for a whale vocalisation, 0 for noise."""
    bitrate = "50k"
    offset = -1
    window = 4

    for i, sample in enumerate(samples):
        print(f"{i}/{len(samples)-1}")
        try:
            inname = correct_path(sample["flac_compressed_xwav_object"])

            begin_extract = max(sample["begin_seek"] + offset, 0)
            end_extract = begin_extract + window

            if (end_extract - begin_extract) < window:
                print(f"Extraction window too short {sample['index']}:{inname.name}")
                continue

            outname = f"{destination}/{sample['label']}_{seed+i:04d}.wav"

            (
                ff.input(inname, ss=begin_extract, to=end_extract)
                .output(outname, format="wav", audio_bitrate=bitrate)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )

            if registry:
                row = [
                    Path(outname).name,
                    sample["index"],
                    sample["label"],
                    Path(sample["flac_compressed_xwav_object"]).name,
                    begin_extract,
                    end_extract,
                    default_count,
                ]
                registry.writerow(row)

        except Exception as e:
            print(f"Error extracting {sample['index']}:{inname.name}")
            print(e)

    return i


# %%
samples = df_mn.to_dicts()

output_path = "dataset"
destination = Path(output_path).resolve()
destination.mkdir(parents=True, exist_ok=True)

registry = "annotations.csv"
fields = ["extraction", "pifsc_index", "label", "source", "begin_seek", "end_seek", "count"]
with open(f"{destination}/{registry}", "a") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    last_index = extract(samples, destination, registry=writer, default_count = 1)

# %%
samples = df_noise.to_dicts()
with open(f"{destination}/{registry}", "a") as f:
    writer = csv.writer(f)
    last_index = extract(samples, destination, seed=last_index, registry=writer)

# %%
df_mn[-10:]
