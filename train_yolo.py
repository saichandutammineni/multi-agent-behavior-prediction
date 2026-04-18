import os

# Fix for Mac OMP Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
import yaml
import random
import numpy as np
import math
from ultralytics import YOLO
from tqdm import tqdm
from data import ParkingProtocol

# --- Configuration ---
DATASET_ROOT = "/Users/sujkrish/Desktop/UCI/Fall_2025/271P_AI/project/Parking-Environment-Behaviour-Prediction/bev-dataset"
YOLO_ROOT = "./yolo_dataset"
IMG_SIZE = 1024
BATCH_SIZE = 8
EPOCHS = 5
MODEL_TYPE = "yolov8n-obb.pt"


def poly_from_rotated_rect(cx, cy, w, h, theta):
    dx = w / 2
    dy = h / 2
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    c, s = math.cos(theta), math.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rotated_corners = np.dot(corners, R.T)
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy
    return rotated_corners.flatten()


def prepare_yolo_dataset():
    if os.path.exists(YOLO_ROOT):
        print(f"--- Recreating Dataset at {YOLO_ROOT} to ensure purity ---")
        shutil.rmtree(YOLO_ROOT)

    print(f"--- Preparing Dataset in {YOLO_ROOT} ---")

    protocol = ParkingProtocol(DATASET_ROOT, mode='detection')
    if len(protocol) == 0: raise ValueError("No frames found!")

    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_ROOT, 'labels', split), exist_ok=True)

    # Deduplicate
    unique_map = {f.image_path: f for f in protocol.items}
    all_frames = list(unique_map.values())

    random.seed(42)
    random.shuffle(all_frames)

    split_idx = int(len(all_frames) * 0.8)
    frames_split = {'train': all_frames[:split_idx], 'val': all_frames[split_idx:]}

    corrupt_count = 0

    for split, frames in frames_split.items():
        for frame in tqdm(frames, desc=f"Processing {split}"):
            safe_name = f"{frame.scene_id}_{frame.frame_id}"

            src_img = frame.image_path
            dst_img = os.path.join(YOLO_ROOT, 'images', split, safe_name + ".png")
            label_path = os.path.join(YOLO_ROOT, 'labels', split, safe_name + ".txt")

            # Copy Image
            shutil.copy(src_img, dst_img)

            valid_agents = 0
            with open(label_path, 'w') as f:
                for agent in frame.meta['agents']:
                    cls_id = 1 if agent['type'] == 'Pedestrian' else 0

                    corners = poly_from_rotated_rect(agent['x'], agent['y'], agent['w'], agent['h'], agent['angle'])

                    # --- CRITICAL FIX: Check for NaN/Inf ---
                    if any(math.isnan(x) or math.isinf(x) for x in corners):
                        continue  # Skip bad object

                    # Clamp to ensure 0-1
                    corners = [min(max(x, 0.0), 1.0) for x in corners]

                    coords = " ".join([f"{x:.6f}" for x in corners])
                    f.write(f"{cls_id} {coords}\n")
                    valid_agents += 1

            # --- CRITICAL FIX: Self-Cleaning ---
            # If no valid agents, or file is weird, verify it
            if valid_agents == 0:
                # Optionally delete empty images if you don't want backgrounds
                # os.remove(label_path)
                # os.remove(dst_img)
                pass

    print(f"Dataset Prep Complete. Found {corrupt_count} corrupt items (skipped).")

    yaml_content = {
        'path': os.path.abspath(YOLO_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'Car', 1: 'Pedestrian'}
    }

    yaml_path = os.path.join(YOLO_ROOT, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    return yaml_path


def train_yolo():
    yaml_path = prepare_yolo_dataset()

    print(f"Loading {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE)

    print("Starting Training...")
    # Using default augmentations but with clean data
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device="mps",
        batch=4,
        workers=4,
        project="parking_lot_yolo",
        name="run_03",

        exist_ok=True,
        # --- CRITICAL MATH FIXES ---
        amp=False,  # Disable Mixed Precision (Keep this false)
        half=False,  # Force full precision

        # --- DISABLE AUGMENTATIONS (The root cause of matmul errors) ---
        mosaic=0.0,  # Disable image mosaic (stitching)
        degrees=0.0,  # Disable random rotation
        shear=0.0,  # Disable shear
        perspective=0.0 # Disable perspective#
    )


if __name__ == "__main__":
    train_yolo()