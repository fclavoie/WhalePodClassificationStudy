"""
This module extracts metadata from .x.flac files, used with custom .x.wav 'harp' chunks.
This format is used by the Pacific Islands Fisheries Science Center (PIFSC) for their passive acoustic monitoring (PAM) data.
The MATLAB code from the Triton repository (https://github.com/MarineBioAcousticsRC/Triton) has been used to understand how to read the .wav file header.

The metadata is extracted from the 'harp' chunk in the .wav file, and saved to a JSON file with the same name as the input FLAC file.
This allows reading the subchunk directly from the .x.flac file, as the subchunks are not necessarily consecutive in the audio file.
"""

import json
import struct
import subprocess
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path


def read_harp_chunk_meta(file_path):
    """Extracts the size and offset of the 'harp' chunk in a .wav file.

    :param file_path: The path to the .wav file.
    :return: A tuple containing the size and offset of the 'harp' chunk, or None if not found.
    """
    with open(file_path, "rb") as f:
        riff = f.read(4).decode("ascii")
        if riff != "RIFF":
            raise ValueError("Invalid file!")
        struct.unpack("<I", f.read(4))[0]
        wave = f.read(4).decode("ascii")
        if wave != "WAVE":
            raise ValueError("Invalid file!")
        while True:
            header = f.read(8)
            if not header:
                break
            chunk_id = header[:4].decode("ascii")
            chunk_size = struct.unpack("<I", header[4:])[0]
            chunk_offset = f.tell() - 8
            if chunk_id == "harp":
                return chunk_size, chunk_offset
            f.seek(chunk_size, 1)
        return None


def read_harp_chunk(file_path, harp_size, harp_offset):
    """Reads the HARP chunk from a .wav file.

    :param file_path: The path to the .wav file.
    :param harp_size: The size of the HARP chunk.
    :param harp_offset: The offset of the HARP chunk.
    :return: The data contained in the HARP chunk.
    """
    with open(file_path, "rb") as f:
        f.seek(harp_offset)
        chunk_id = f.read(4).decode("ascii")
        if chunk_id != "harp":
            raise ValueError(f"Chunk found, but not is not 'harp': {chunk_id}")
        chunk_size = struct.unpack("<I", f.read(4))[0]
        if chunk_size != harp_size:
            raise ValueError(f"Unexpexted HARP chunk size: {chunk_size}")
        version = struct.unpack("<B", f.read(1))[0]
        firmware_version = f.read(10).decode("ascii").strip(" \x00")
        instrument_id = f.read(4).decode("ascii").strip()
        site_name = f.read(4).decode("ascii").strip()
        experiment_name = f.read(8).decode("ascii").strip()
        disk_sequence = struct.unpack("<B", f.read(1))[0]  # noqa: F841
        disk_serial_number = f.read(8).decode("ascii").strip()  # noqa: F841
        num_raw_files = struct.unpack("<H", f.read(2))[0]
        longitude = struct.unpack("<i", f.read(4))[0]  # noqa: F841
        latitude = struct.unpack("<i", f.read(4))[0]  # noqa: F841
        depth = struct.unpack("<h", f.read(2))[0]  # noqa: F841
        f.read(8)
        expected_harp_subchunk_size = 64 - 8 + num_raw_files * 32
        if chunk_size != expected_harp_subchunk_size:
            raise ValueError(f"Unexpected HARP subchunk size: {chunk_size}")
        raw_files = []
        for _ in range(num_raw_files):
            year = struct.unpack("<B", f.read(1))[0] + 2000
            month = struct.unpack("<B", f.read(1))[0]
            day = struct.unpack("<B", f.read(1))[0]
            hour = struct.unpack("<B", f.read(1))[0]
            minute = struct.unpack("<B", f.read(1))[0]
            second = struct.unpack("<B", f.read(1))[0]
            ticks = struct.unpack("<H", f.read(2))[0]
            byte_loc = struct.unpack("<I", f.read(4))[0]
            byte_length = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<I", f.read(4))[0]
            sample_rate = struct.unpack("<I", f.read(4))[0]
            gain = struct.unpack("<B", f.read(1))[0]
            f.read(7)
            start_time = datetime(year, month, day, hour, minute, second) + timedelta(
                milliseconds=ticks
            )
            raw_files.append(
                {
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "byte_loc": byte_loc,
                    "byte_length": byte_length,
                    "sample_rate": sample_rate,
                    "gain": gain,
                }
            )
        return {
            "version": version,
            "firmware_version": firmware_version,
            "instrument_id": instrument_id,
            "site_name": site_name,
            "experiment_name": experiment_name,
            "num_raw_files": num_raw_files,
            "raw_files": raw_files,
        }


def flac_to_wav(input_path, output_wav):
    """Converts a FLAC file to a WAV file.

    :param input_path: The path to the input FLAC file.
    :param output_wav: The path to the output WAV file.
    """
    command = [
        "flac",
        "-df",
        "--preserve-modtime",
        "--keep-foreign-metadata",
        "-o",
        output_wav,
        input_path,
    ]
    subprocess.run(command, check=True)


def process_file(task):
    """Processes a FLAC file by converting it to WAV, extracting 'harp' chunk metadata, and saving the metadata to a JSON file.

    :param task: A tuple containing the path to the input FLAC file and the output directory.
    """
    input_file, output_dir = task
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    wav_file = output_dir / (input_file.stem + ".wav")
    json_file = output_dir / (input_file.stem + ".json")
    try:
        flac_to_wav(str(input_file), str(wav_file))
        harp_size, harp_offset = read_harp_chunk_meta(str(wav_file))
        harp_data = read_harp_chunk(
            str(wav_file), harp_size=harp_size, harp_offset=harp_offset
        )
        if harp_data:
            harp_data["filename"] = input_file.name
            with open(json_file, "w") as f:
                json.dump(harp_data, f, indent=4)
        print(f"Processing succceded : {input_file}")
    except Exception as e:
        print(f"Error while processing {input_file} : {e}")
    finally:
        if wav_file.exists():
            wav_file.unlink()


def process_directory(input_dir, output_dir, num_workers=None):
    """Processes all FLAC files in a directory by converting them to WAV, extracting 'harp' chunk metadata, and saving the metadata to JSON files.

    :param input_dir: The path to the input directory containing the FLAC files.
    :param output_dir: The path to the output directory where the JSON files will be saved.
    :param num_workers: The number of worker processes to use for parallel processing.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flac_files = list(input_dir.rglob("*.flac"))
    tasks = [(str(flac_file), str(output_dir)) for flac_file in flac_files]
    num_workers = num_workers or cpu_count()
    with Pool(num_workers) as pool:
        pool.map(process_file, tasks)


process_directory(
    "../pifsc/audio/hawaii/pipan_hawaii_30/audio",
    "../pifsc/audio/hawaii/meta",
    num_workers=8,
)
