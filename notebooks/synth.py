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
# # Synthetic data generation
#
# This notebook can be use to generate synthesized audio file, i.e. overlaying files with a single whale vocalisation to have several of them in the same file.
# The is also an option to add noises from other sources.

# %%
import random
from pathlib import Path

import polars as pl
from pydub import AudioSegment

# %%
folder = Path().resolve().parent
print(folder)

infolder = folder / "dataset"
outfolder = folder / "dataset_synth"

# %%
df = pl.read_csv(infolder / "annotations.csv")


# %%
def synthesize(input_filenames: list, output_filename: str):
    """ Synthesize a new audio file by overlaying multiple audio files. """
    overlay = AudioSegment.from_file(f"{infolder}/{input_filenames[0]}", format="wav")
    
    for filename in input_filenames[1:]:
        track = AudioSegment.from_file(f"{infolder}/{filename}", format="wav")
        overlay = overlay.overlay(track)
    
    overlay.export(f"{outfolder}/{output_filename}", format="wav")


# %%
df_target = df.filter(pl.col("label") == "Mn").select(["filename", "count"])

# %%
df_nontarget = df.filter(pl.col("label") != "Mn").select(["filename", "count"])

# %%
add_noise = True

output_filenames = []
target_counts = []
nontarget_counts = []
extraction_lists = []

min_sample_count = 4
max_sample_count = 6

for i in range(2000):
    print(f"{i+1}/2000")
    
    total_extractions = random.randint(min_sample_count, max_sample_count)
    
    sample_count = random.randint(0, total_extractions)
    sample = df_target.sample(n=sample_count, shuffle=True)
    target_count = sample["count"].sum()
    
    if add_noise:
        nontarget_count = total_extractions - sample_count
    elif not sample_count:
        nontarget_count = random.randint(1, max_sample_count)
    else:
        nontarget_count = 0
    
    neg_sample = df_nontarget.sample(n=nontarget_count, shuffle=True)
    
    extractions = sample["filename"].to_list() + neg_sample["filename"].to_list()
    
    output_filename = f"{str(i).zfill(4)}.wav"
    synthesize(extractions, output_filename)
    
    output_filenames.append(output_filename)
    target_counts.append(target_count)
    nontarget_counts.append(nontarget_count)
    extraction_lists.append(",".join(extractions))

df_synth = pl.DataFrame({
    "filename": output_filenames,
    "target": target_counts,
    "nontarget": nontarget_counts,
    "augmentations": "mixup",
    "extractions": extraction_lists
})

# %%
df_synth.write_csv(f"{outfolder}/annotations.csv")
