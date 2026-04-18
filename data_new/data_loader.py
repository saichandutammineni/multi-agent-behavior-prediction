import os
import json
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict


class Frame:
    """Represents a single image frame."""

    def __init__(self, image_path: str, frame_idx: int):
        self.image_path = image_path
        self.frame_idx = frame_idx

    def load_image(self) -> np.ndarray:
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        return cv2.imread(self.image_path)


class Sequence:
    """Represents a 10s (approx) continuous clip."""

    def __init__(self, seq_id: str, frames: List[Frame]):
        self.seq_id = seq_id
        # Ensure temporal order
        self.frames = sorted(frames, key=lambda f: f.frame_idx)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx) -> Frame:
        return self.frames[idx]


class Scene:
    """Represents a Scene Folder containing multiple sequences."""

    def __init__(self, scene_path: str):
        self.name = os.path.basename(scene_path)
        self.seq_dir = os.path.join(scene_path, "sequences")
        self.sequences = []
        self._load_sequences()

    def _load_sequences(self):
        # Find JSONs: sequences/seq_00/seq.json
        seq_json_files = sorted(glob.glob(os.path.join(self.seq_dir, "**", "*.json"), recursive=True))

        for seq_file in seq_json_files:
            try:
                with open(seq_file, 'r') as f:
                    content = json.load(f)
            except:
                continue

            seq_frames_data = content.get('frames', []) if isinstance(content, dict) else content
            frame_objs = []
            base_dir = os.path.dirname(seq_file)

            for item in seq_frames_data:
                # Resolve image path
                abs_img = os.path.abspath(os.path.join(base_dir, item['image']))
                if os.path.exists(abs_img):
                    f_obj = Frame(abs_img, frame_idx=item.get('frame_idx', -1))
                    frame_objs.append(f_obj)

            if frame_objs:
                seq_id = os.path.basename(os.path.dirname(seq_file))
                self.sequences.append(Sequence(seq_id, frame_objs))


class ParkingSequenceLoader(Dataset):
    """
    Yields Sequence objects (lists of Frames) for Tracking/LSTM generation.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.sequences = []
        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            print(f"[Error] {self.root_dir} not found.")
            return

        for item in sorted(os.listdir(self.root_dir)):
            path = os.path.join(self.root_dir, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "sequences")):
                scene = Scene(path)
                self.sequences.extend(scene.sequences)

        print(f"[Loader] Loaded {len(self.sequences)} total sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]