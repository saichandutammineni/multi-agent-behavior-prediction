import os
from pathlib import Path


def filter_annotations(root_path, step=25):
    # Expand the ~ in the path to the full user directory
    base_dir = Path(root_path).expanduser()

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    # Find all Scene directories (DJI_XXXX)
    scene_dirs = sorted([d for d in base_dir.glob("DJI_*") if d.is_dir()])

    if not scene_dirs:
        print(f"No 'DJI_XXXX' directories found in {base_dir}")
        return

    total_deleted = 0
    total_kept = 0

    print(f"Found {len(scene_dirs)} scenes. Starting cleanup of annotations...\n")

    for scene_dir in scene_dirs:
        annot_dir = scene_dir / "annotations"

        if not annot_dir.exists():
            # print(f"  [!] No 'annotations' folder in {scene_dir.name}")
            continue

        scene_deleted = 0
        scene_kept = 0

        # Iterate over all .json files in the annotations directory
        for json_file in annot_dir.glob("*.json"):
            filename = json_file.name

            try:
                # Expecting format: "000123_tokenid.json"
                # Split by '_' to get the number part ("000123")
                frame_idx_str = filename.split('_')[0]

                # Convert to integer
                frame_idx = int(frame_idx_str)

                # Check if it matches the sampling step (divisible by 25)
                if frame_idx % step == 0:
                    # KEEP this file
                    scene_kept += 1
                else:
                    # DELETE this file
                    os.remove(json_file)
                    scene_deleted += 1

            except ValueError:
                # If filename doesn't start with a number (e.g., "meta.json"), skip it safely
                print(f"    [?] Skipping non-standard file: {filename}")
                continue

        print(f"  {scene_dir.name}: Kept {scene_kept}, Deleted {scene_deleted}")
        total_kept += scene_kept
        total_deleted += scene_deleted

    print(f"\nCompleted! Total files kept: {total_kept}, Total files deleted: {total_deleted}")


if __name__ == "__main__":
    # Using the path you provided (handling the ~ automatically)
    ROOT_PATH = "~/Desktop/UCI/Fall_2025/271P_AI/project/dlp-dataset/dlp/bev-dataset"

    # Run the filter
    filter_annotations(ROOT_PATH)