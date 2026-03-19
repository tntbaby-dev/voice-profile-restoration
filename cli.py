from pathlib import Path
import argparse

from run_batch import process_one_file
from voice_profile_restoration.analyzers.profile_builder import build_speaker_profile


def main():
    parser = argparse.ArgumentParser(description="Batch Voice Restoration Tool")

    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="List of input folders containing WAV files",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory",
    )

    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    output_dir = Path(args.output_dir)

    restored_dir = output_dir / "restored"
    polished_dir = output_dir / "polished"

    restored_dir.mkdir(parents=True, exist_ok=True)
    polished_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 your reference files (keep same as before)
    reference_files = [
        "/Users/mac/Desktop/BRANDING/Sound_Design/Voice_Training_Data_Blueprint/training_data/grace_rap.wav",
        "/Users/mac/Desktop/BRANDING/Sound_Design/Voice_Training_Data_Blueprint/training_data/fast_rap_001.wav",
        "/Users/mac/Desktop/BRANDING/Sound_Design/Voice_Training_Data_Blueprint/training_data/slow_rap_002.wav",
    ]

    reference_profile = build_speaker_profile(reference_files)

    wav_files = []
    for folder in input_dirs:
        wav_files.extend(sorted(folder.glob("*.wav")))

    print(f"Found {len(wav_files)} files.")

    for i, wav in enumerate(wav_files, start=1):
        print(f"[{i}/{len(wav_files)}] Processing {wav.name}")

        process_one_file(
            wav,
            reference_profile,
            iterations=3,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()