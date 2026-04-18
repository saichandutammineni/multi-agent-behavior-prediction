import os
from pathlib import Path


def replace_sequence_files(root_path):
    base_dir = Path(root_path).expanduser()

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    # 1. Find all Scene directories (DJI_00XX)
    scene_dirs = sorted([d for d in base_dir.glob("DJI_*") if d.is_dir()])

    if not scene_dirs:
        print(f"No 'DJI_XXXX' directories found in {base_dir}")
        return

    count_replaced = 0

    for scene_dir in scene_dirs:
        print(f"Scanning Scene: {scene_dir.name}")

        # 2. Go into /sequences
        sequences_path = scene_dir / "sequences"
        if not sequences_path.exists():
            continue

        # 3. Find all sequence directories (seq_XXXXXX)
        seq_dirs = sorted([d for d in sequences_path.glob("seq_*") if d.is_dir()])

        for seq_dir in seq_dirs:
            original_file = seq_dir / "sequence.json"
            sampled_file = seq_dir / "sampled_sequence.json"

            # STRICT SAFETY CHECK: Only proceed if sampled_sequence.json exists
            if sampled_file.exists():
                try:
                    # Step A: Delete the old sequence.json if it exists
                    if original_file.exists():
                        os.remove(original_file)

                    # Step B: Rename sampled_sequence.json -> sequence.json
                    sampled_file.rename(original_file)

                    # print(f"  [✓] Replaced in {seq_dir.name}")
                    count_replaced += 1

                except OSError as e:
                    print(f"  [X] Error in {seq_dir.name}: {e}")
            else:
                # Optional: Warn if sampled file is missing
                # print(f"  [!] Skipping {seq_dir.name}: No sampled_sequence.json found.")
                pass

    print(f"\nSuccess! Updated {count_replaced} sequence files.")


if __name__ == "__main__":
    # The full path you provided
    ROOT_PATH = "/Users/sujkrish/Desktop/UCI/Fall_2025/271P_AI/project/dlp-dataset/dlp/bev-dataset"

    replace_sequence_files(ROOT_PATH)