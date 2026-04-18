import os
from pathlib import Path


def filter_images(root_path, step=25):
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

    print(f"Found {len(scene_dirs)} scenes. Starting cleanup of images...\n")

    for scene_dir in scene_dirs:
        # Target the 'images' directory instead of annotations
        img_dir = scene_dir / "images"

        if not img_dir.exists():
            # print(f"  [!] No 'images' folder in {scene_dir.name}")
            continue

        scene_deleted = 0
        scene_kept = 0

        # Iterate over all .png files in the images directory
        for img_file in img_dir.glob("*.png"):
            filename = img_file.name

            try:
                # Expecting format: "000123_tokenid.png"
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
                    os.remove(img_file)
                    scene_deleted += 1

            except ValueError:
                # If filename doesn't start with a number, skip it safely
                print(f"    [?] Skipping non-standard file: {filename}")
                continue

        print(f"  {scene_dir.name}: Kept {scene_kept}, Deleted {scene_deleted}")
        total_kept += scene_kept
        total_deleted += scene_deleted

    print(f"\nCompleted! Total images kept: {total_kept}, Total images deleted: {total_deleted}")


if __name__ == "__main__":
    # Using the path provided
    ROOT_PATH = "~/Desktop/UCI/Fall_2025/271P_AI/project/dlp-dataset/dlp/bev-dataset"

    # Run the filter
    filter_images(ROOT_PATH)