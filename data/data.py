import os
import json
import glob
import math
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict


class Frame:
    def __init__(self, scene_id: str, image_path: str, json_path: str, frame_idx: int = -1):
        self.scene_id = scene_id
        self.image_path = image_path
        self.json_path = json_path
        self.frame_id = os.path.splitext(os.path.basename(json_path))[0]
        self.frame_idx = frame_idx
        self._meta_cache = None

    @property
    def meta(self) -> Dict:
        if self._meta_cache is None:
            if not os.path.exists(self.json_path):
                return {"meta": {}, "agents": []}

            with open(self.json_path, 'r') as f:
                data = json.load(f)

            bounds = data.get('bbox_world', {})
            x_min = bounds.get('xmin', 0)
            x_max = bounds.get('xmax', 1)
            y_min = bounds.get('ymin', 0)
            y_max = bounds.get('ymax', 1)

            map_w = x_max - x_min
            map_h = y_max - y_min

            # 1. Safety: If map is zero-size, we can't normalize.
            if map_w <= 0.1 or map_h <= 0.1:
                return {"meta": data, "agents": []}

            def process_obj(obj, is_static=False):
                try:
                    # Extract Data
                    if is_static:
                        wx, wy = obj['position'][0], obj['position'][1]
                        l, w = obj['size'][0], obj['size'][1]
                    else:
                        wx, wy = obj['center']['x'], obj['center']['y']
                        l, w = obj['size']['length'], obj['size']['width']

                    # 2. Safety: Handle None or Zero dimensions
                    if l is None or w is None or l <= 0 or w <= 0:
                        return None

                    # Normalize Position
                    cx = (wx - x_min) / map_w
                    cy = 1.0 - ((wy - y_min) / map_h)

                    # Normalize Size
                    nw = w / map_w
                    nl = l / map_h

                    # 3. Safety: Out of Bounds Check
                    # If object is > 50% outside the image, ignore it to prevent math errors
                    if not (-0.5 <= cx <= 1.5 and -0.5 <= cy <= 1.5):
                        return None

                    # 4. Safety: Handle missing Heading
                    heading = obj.get('heading', 0)
                    if heading is None: heading = 0
                    angle = -heading

                    return {
                        "type": obj.get('type', 'Car'),
                        "x": cx, "y": cy,
                        "w": nl, "h": nw,
                        "angle": angle
                    }
                except Exception:
                    return None

            unified_agents = []
            for obj in data.get('static_objects', []):
                res = process_obj(obj, is_static=True)
                if res: unified_agents.append(res)

            for obj in data.get('agents', []):
                res = process_obj(obj, is_static=False)
                if res: unified_agents.append(res)

            self._meta_cache = {
                "meta": data,
                "agents" : unified_agents
            }

        return self._meta_cache


class Scene:
    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.name = os.path.basename(scene_path)
        self.img_dir = os.path.join(scene_path, "images")
        self.ann_dir = os.path.join(scene_path, "annotations")
        self.all_frames = []
        self._load_frames()

    def _load_frames(self):
        if not os.path.exists(self.img_dir): return
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.img_dir, ext)))

        for img_path in sorted(image_files):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(self.ann_dir, basename + ".json")
            if os.path.exists(json_path):
                self.all_frames.append(Frame(self.name, img_path, json_path))


class ParkingProtocol(Dataset):
    def __init__(self, root_dir: str, mode: str = 'detection'):
        self.root_dir = root_dir
        self.mode = mode
        self.items = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"[Loader] Init in mode='{self.mode}'...")
        if not os.path.exists(self.root_dir): return
        for item in sorted(os.listdir(self.root_dir)):
            path = os.path.join(self.root_dir, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "annotations")):
                scene = Scene(path)
                if self.mode == 'detection':
                    self.items.extend(scene.all_frames)
        print(f"[Loader] Loaded {len(self.items)} items ({self.mode}).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]