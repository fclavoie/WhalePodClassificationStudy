"""
This module contains functions for augmenting audio data.
It includes functions to add Gaussian noise, pitch shift, and apply frequency masking to audio waveforms.

Configurations:
    input_dir: Directory containing the input dataset.
    output_dir: Directory to save the augmented dataset.
    SAMPLE_RATE: Sample rate for audio processing.
    MAX_MIXUP: Maximum number of additional files for mixup.
    EMPTY_COUNT: Number of empty samples to generate.
    SEED: Random seed for reproducibility.
    GENERATION_FACTOR: Multiplier to determine the total number of augmented files.
"""

import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import polars as pl
from tqdm import tqdm


input_dir = Path("../dataset")
output_dir = Path("../dataset_aug")
output_dir.mkdir(parents=True, exist_ok=True)

EMPTY_COUNT = 100
GENERATION_FACTOR = 4
MAX_MIXUP = 4
SAMPLE_RATE = 10000
SEED = 42
random.seed(SEED)

def add_gaussian_noise(waveform, noise_level=0.005):
    """Adds Gaussian noise to the waveform.

    :param waveform: The input audio waveform.
    :param noise_level: The standard deviation of the Gaussian noise, defaults to 0.005
    :return: The waveform with added Gaussian noise.
    """
    noise = noise_level * torch.randn_like(waveform)
    return waveform + noise


def pitch_shift(waveform, sample_rate, n_steps):
    """Shifts the pitch of the waveform.

    :param waveform: The input audio waveform.
    :param sample_rate: The sample rate of the audio.
    :param n_steps: The number of steps to shift the pitch.
    :return: The pitch-shifted waveform.
    """
    return T.PitchShift(sample_rate=sample_rate, n_steps=n_steps)(waveform)


def frequency_mask(waveform, freq_mask_param=15):
    """Applies a frequency mask to the waveform.

    :param waveform: The input audio waveform.
    :param freq_mask_param: The maximum width of the frequency mask
    :return: The waveform with the frequency mask applied.
    """
    n_fft = 512
    hop_length = 128
    win_length = n_fft

    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=None
    )
    spectrogram = spectrogram_transform(waveform)

    magnitude_spectrogram = torch.abs(spectrogram)

    freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    masked_spectrogram = freq_mask(magnitude_spectrogram)

    griffin_lim = T.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    reconstructed_waveform = griffin_lim(masked_spectrogram)

    return reconstructed_waveform


def mixup_samples(waveforms):
    """Mixes multiple waveforms together by averaging them.

    :param waveforms: A list of audio waveforms to be mixed.
    :return: The mixed waveform.
    """
    mixup = waveforms[0]
    for waveform in waveforms[1:]:
        mixup += waveform
    return mixup / len(waveforms)


def apply_augmentations(waveform, augmentations, max_augmentations=4):
    """Applies a random selection of augmentations to the waveform.

    :param waveform: The input audio waveform.
    :param augmentations: A dictionary of augmentation functions.
    :param max_augmentations: The maximum number of augmentations to apply.
    :return: The augmented waveform and a list of applied augmentations.
    """
    selected_augmentations = random.sample(
        list(augmentations.items()), k=random.randint(1, max_augmentations)
    )
    augmented_waveform = waveform.clone()
    applied_augmentations = []

    for aug_name, aug_func in selected_augmentations:
        # Ensure pitch_shift_up and pitch_shift_down are not applied together
        if "pitch_shift" in applied_augmentations and "pitch_shift" in aug_name:
            continue
        augmented_waveform = aug_func(augmented_waveform)
        applied_augmentations.append(aug_name)

    return augmented_waveform, applied_augmentations


annotations = pl.read_csv(input_dir / "annotations.csv")
original_files = annotations.filter(
    (pl.col("pifsc_index").is_null()) & (pl.col("label") == "Mn")
)
mixup_candidates = annotations.filter(pl.col("label").is_in(["Background", "Other"]))

output_filenames = []
target_counts = []
nontarget_counts = []
extraction_lists = []
augmentation_lists = []

# Copy samples
file_index = 0
for file in tqdm(original_files["filename"].to_list(), desc="Copying original files"):
    file_path = input_dir / file
    output_filename = f"{file_index:04d}.wav"
    output_path = output_dir / output_filename

    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, SAMPLE_RATE)

    output_filenames.append(output_filename)
    target_counts.append(annotations.filter(pl.col("filename") == file)["count"].sum())
    nontarget_counts.append(0)
    extraction_lists.append([file])
    augmentation_lists.append(["original"])

    file_index += 1

# Copy background noise samples
background_files = (
    mixup_candidates.filter(pl.col("label") == "Background")
    .sample(n=EMPTY_COUNT // 2, seed=SEED)["filename"]
    .to_list()
)
other_files = (
    mixup_candidates.filter(pl.col("label") == "Other")
    .sample(n=EMPTY_COUNT // 2, seed=SEED)["filename"]
    .to_list()
)

for label_files in [background_files, other_files]:
    for file in label_files:
        file_path = input_dir / file
        output_filename = f"{file_index:04d}.wav"
        output_path = output_dir / output_filename

        waveform, sr = torchaudio.load(file_path)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        torchaudio.save(output_path, waveform, SAMPLE_RATE)

        output_filenames.append(output_filename)
        target_counts.append(0)
        nontarget_counts.append(1)
        extraction_lists.append([file])
        augmentation_lists.append(["original"])

        file_index += 1

augmentations = {
    "gaussian_noise": lambda wf: add_gaussian_noise(wf),
    "pitch_shift_up": lambda wf: pitch_shift(wf, SAMPLE_RATE, n_steps=2),
    "pitch_shift_down": lambda wf: pitch_shift(wf, SAMPLE_RATE, n_steps=-2),
    "frequency_mask": lambda wf: frequency_mask(wf),
}

# Generate augmented samples
for file in tqdm(original_files["filename"].to_list(), desc="Creating augmented files"):
    for _ in range(GENERATION_FACTOR):
        file_path = input_dir / file

        waveform, sr = torchaudio.load(file_path)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        applied_extractions = [file]
        applied_augmentations = []

        # Apply mixup
        if random.random() < 0.5:
            mixup_files = random.sample(
                mixup_candidates["filename"].to_list(), k=random.randint(1, MAX_MIXUP)
            )
            mixup_waveforms = [
                torchaudio.load(input_dir / mixup_file)[0] for mixup_file in mixup_files
            ]
            waveform = mixup_samples([waveform] + mixup_waveforms)
            applied_augmentations.append("mixup")
            applied_extractions.extend(mixup_files)

        # Apply random augmentations
        augmented_waveform, random_augmentations = apply_augmentations(
            waveform, augmentations
        )
        applied_augmentations += random_augmentations

        output_filename = f"{file_index:04d}.wav"
        output_path = output_dir / output_filename
        torchaudio.save(output_path, augmented_waveform, SAMPLE_RATE)

        output_filenames.append(output_filename)
        target_counts.append(
            annotations.filter(pl.col("filename") == file)["count"].sum()
        )
        nontarget_counts.append(
            len(applied_extractions) - 1
        )
        extraction_lists.append(applied_extractions)
        augmentation_lists.append(applied_augmentations)

        file_index += 1

# Save annotations
output_annotations = pl.DataFrame(
    {
        "filename": output_filenames,
        "target": target_counts,
        "nontarget": nontarget_counts,
        "augmentations": [
            "|".join(aug) if isinstance(aug, list) else aug
            for aug in augmentation_lists
        ],
        "extractions": [
            "|".join(ext) if isinstance(ext, list) else ext for ext in extraction_lists
        ],
    }
)

output_annotations.write_csv(output_dir / "annotations.csv")
