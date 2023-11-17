#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from pathlib import Path

import torch

ALLOWED_AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg"]
SAMPLING_RATE = 16000


def is_audio_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in ALLOWED_AUDIO_EXTENSIONS


def downsample(file_path):
    new_filename = "output/" + Path(file_path).stem + "_processed.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-ar", str(SAMPLING_RATE), new_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return new_filename


def extract_voice(file_path):
    try:
        if not is_audio_file(file_path):
            raise ValueError(f"{file_path} is not a valid audio file.")

        target_file = downsample(file_path)

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            _,
            collect_chunks,
        ) = utils

        s = time.time()

        wav = read_audio(target_file, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            wav, model, sampling_rate=SAMPLING_RATE
        )
        save_audio(
            target_file,
            collect_chunks(speech_timestamps, wav),
            sampling_rate=SAMPLING_RATE,
        )

        e = time.time()
        print(f"\n🎧 Extracted file: {target_file}\n🚀 Execution time: {e - s} seconds")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract voice from an audio file.")
    parser.add_argument(
        "-f", "--file", help="Path to the file to be processed.", required=True
    )

    args = parser.parse_args()
    extract_voice(args.file)


if __name__ == "__main__":
    main()
