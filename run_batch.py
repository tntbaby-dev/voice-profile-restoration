from __future__ import annotations

from pathlib import Path
import json
import csv


from voice_profile_restoration.analyzers.quality_scorer import compute_quality_score 
from voice_profile_restoration.analyzers.profile_builder import build_speaker_profile
from voice_profile_restoration.analyzers.spectral_features import extract_spectral_features
from voice_profile_restoration.analyzers.imbalance_analyzer import compute_imbalance
from voice_profile_restoration.analyzers.eq_mapper import map_to_eq_settings
from voice_profile_restoration.analyzers.eq_processor import process_file
from voice_profile_restoration.analyzers.polish_processor import process_polish_file


REFERENCE_FILES = [
    "/Users/mac/Desktop/BRANDING/Sound_Design/Voice_Training_Data_Blueprint/training_data/grace_rap.wav",
    "/Users/mac/Desktop/BRANDING/Sound_Design/Voice_Training_Data_Blueprint/training_data/fast_rap_001.wav",
    "/Users/mac/Desktop/BRANDING/Sound_Design/Voice_Training_Data_Blueprint/training_data/slow_rap_002.wav",
]

INPUT_DIRS = [
    Path("/Users/mac/Desktop/BRANDING/Sound_Design/Phone_Recording_A"),
    Path("/Users/mac/Desktop/BRANDING/Sound_Design/Phone_Recording_B"),
    Path("/Users/mac/Desktop/BRANDING/Sound_Design/Phone_Recording_C"),
]

RESTORED_DIR = Path("outputs/restored")
POLISHED_DIR = Path("outputs/polished")
REPORT_PATH = Path("reports/batch_report.json")


def process_one_file(input_file: Path, reference_profile: dict, iterations: int = 3) -> dict:
    current_file = input_file
    iteration_reports = []

    for i in range(iterations):
        features = extract_spectral_features(current_file)
        imbalance = compute_imbalance(features, reference_profile)
        eq = map_to_eq_settings(imbalance)

        restored_file = RESTORED_DIR / f"{input_file.stem}_restored_step_{i+1}.wav"
        process_file(current_file, restored_file, eq)

        iteration_reports.append(
            {
                "iteration": i + 1,
                "features": features,
                "imbalance": imbalance,
                "eq_settings": eq,
                "output_file": str(restored_file),
            }
        )

        current_file = restored_file

    polished_file = POLISHED_DIR / f"{input_file.stem}_polished.wav"
    process_polish_file(current_file, polished_file)

    final_features = extract_spectral_features(polished_file)

    before_features = extract_spectral_features(input_file)
    before_score = compute_quality_score(before_features, reference_profile)

    after_score = compute_quality_score(final_features, reference_profile)
    improvement = after_score - before_score

    return {
        "input_file": str(input_file),
        "iterations": iteration_reports,
        "final_output_file": str(polished_file),
        "before_features": before_features,
        "final_features": final_features,
        "before_score": before_score,
        "after_score": after_score,
        "improvement": improvement,
    }


def main() -> None:
    print("DEBUG: main started")
    RESTORED_DIR.mkdir(parents=True, exist_ok=True)
    POLISHED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reference_profile = build_speaker_profile(REFERENCE_FILES)

    wav_files = []
    for folder in INPUT_DIRS:
        wav_files.extend(sorted(folder.glob("*.wav")))

    print(f"DEBUG: found {len(wav_files)} wav files")

    if not wav_files:
        raise ValueError("No WAV files found in the provided folders.")

    full_report = {
        "reference_profile": reference_profile,
        "files_processed": [],
    }

    print(f"Found {len(wav_files)} wav file(s).")

    for idx, wav_file in enumerate(wav_files, start=1):
        print(f"[{idx}/{len(wav_files)}] Processing: {wav_file.name}")
        result = process_one_file(wav_file, reference_profile, iterations=3)
        full_report["files_processed"].append(result)
        print(f"    Done -> {result['final_output_file']}")

        # AFTER the loop finishes
    csv_path = "reports/batch_summary.csv"

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["file", "before_score", "after_score", "improvement"])

        for item in full_report["files_processed"]:
            writer.writerow([
                item["input_file"],
                item["before_score"],
                item["after_score"],
                item["improvement"],
            ])

    print(f"CSV summary saved to: {csv_path}")


    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\nBatch processing complete.")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
        main()