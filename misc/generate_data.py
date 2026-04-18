#!/usr/bin/env python3
"""
generate_bev_dataset.py

Creates Bird's-Eye-View (BEV) images and annotations from parking-lot dataset scenes.
Optimized with multiprocessing and vectorized geometry (DLP-Dataset style).

Outputs per scene:
 - /images        (PNG per frame)
 - /annotations   (JSON per frame, with world-frame 4-corner boxes)
 - /sequences     (10-second sequences)
 - summary.json
"""

import os
import json
import argparse
import math
import glob
import time
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm  # Recommended: pip install tqdm


# ============================================================
# JSON Utility
# ============================================================

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def find_scene_groups(input_dir: str):
    """Detect scenes based on *_scene.json naming."""
    scene_files = glob.glob(os.path.join(input_dir, "*_scene.json"))
    groups = {}

    for sf in scene_files:
        base = os.path.basename(sf).replace("_scene.json", "")
        groups[base] = {
            "scene": sf,
            "agents": os.path.join(input_dir, f"{base}_agents.json"),
            "obstacles": os.path.join(input_dir, f"{base}_obstacles.json"),
            "frames": os.path.join(input_dir, f"{base}_frames.json"),
            "instances": os.path.join(input_dir, f"{base}_instances.json"),
        }

    return groups


# ============================================================
# Extract Position / Heading (Optimized)
# ============================================================

def try_get_position(obj: dict) -> Tuple[float, float]:
    """
    Try many common formats to extract (x,y).
    Optimized order for common DLP/NuScenes formats.
    """
    # Fast path for common formats
    if "translation" in obj:
        v = obj["translation"]
        return float(v[0]), float(v[1])
    if "x" in obj and "y" in obj:
        return float(obj["x"]), float(obj["y"])

    # Slow path for irregular keys
    keys = ["translation_m", "position", "coords", "center", "location", "loc", "pose"]
    for k in keys:
        if k in obj:
            v = obj[k]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return float(v[0]), float(v[1])
            if isinstance(v, dict) and "x" in v and "y" in v:
                return float(v["x"]), float(v["y"])

    raise KeyError(f"No usable position in keys: {obj.keys()}")


def try_get_heading(obj: dict) -> float:
    """Try to extract yaw/heading in radians."""
    # Fast path
    if "heading" in obj:
        return float(obj["heading"])

    # Rotation quaternion
    if "rotation" in obj:
        r = obj["rotation"]
        if isinstance(r, (list, tuple)) and len(r) == 4:
            return quat_to_yaw(r)

    # Fallbacks
    for k in ["yaw", "rot", "phi", "angle", "bearing"]:
        if k in obj:
            val = obj[k]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                return float(val[0])
    return 0.0


def quat_to_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ============================================================
# Vectorized Geometry Helpers
# ============================================================

def get_rotated_corners_vectorized(cx, cy, length, width, heading):
    """
    Return world-frame points of an oriented rectangle using NumPy.
    Returns: np.ndarray shape (4, 2)
    """
    # Rotation matrix
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])

    # Local corners [x, y] relative to center
    # Order: FL, FR, RR, RL
    l2, w2 = length / 2.0, width / 2.0
    corners_local = np.array([
        [l2, w2],
        [l2, -w2],
        [-l2, -w2],
        [-l2, w2]
    ])

    # Rotate and translate
    corners_world = (corners_local @ R.T) + np.array([cx, cy])
    return corners_world


def world_to_pixel_matrix(pts_world, bbox, W, H):
    """
    Vectorized projection of world coordinates to pixels.
    pts_world: (N, 2) array
    bbox: (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = bbox

    # Scale factors
    scale_x = (W - 1) / (xmax - xmin)
    scale_y = (H - 1) / (ymax - ymin)

    # Normalize and scale
    u = (pts_world[:, 0] - xmin) * scale_x
    v = (ymax - pts_world[:, 1]) * scale_y  # Invert Y for image coords

    # Round and cast to int
    pts_px = np.stack([u, v], axis=1)
    return np.rint(pts_px).astype(np.int32)


def draw_rotated_box_fast(canvas, pts_world, bbox, W, H, color, fill=True, thickness=2):
    """Draws using vectorized coordinate transformation."""
    pts_px = world_to_pixel_matrix(pts_world, bbox, W, H)

    # cv2.fillPoly expects a list of arrays
    if fill:
        cv2.fillPoly(canvas, [pts_px], color)
    else:
        cv2.polylines(canvas, [pts_px], True, color, thickness)


# ============================================================
# Unified Color Function
# ============================================================

def color_for_type(obj_type: str):
    t = obj_type.lower()
    if "car" in t or "vehicle" in t:
        return (0, 200, 0)  # green
    if "ped" in t or "person" in t:
        return (0, 0, 255)  # red
    if "bike" in t or "bicycle" in t or "cyclist" in t:
        return (255, 0, 0)  # blue
    return (255, 255, 255)  # default white


# ============================================================
# Process One Scene
# ============================================================

def process_scene(args_pack):
    """
    Worker function for multiprocessing.
    args_pack is a tuple: (base_name, files_dict, out_root, params)
    """
    base, files, out_root, params = args_pack

    try:
        # Load Data
        agents = load_json(files["agents"])
        obstacles = load_json(files["obstacles"])
        frames = load_json(files["frames"])
        instances = load_json(files["instances"])
        # scene_meta = load_json(files["scene"]) # Unused but available

        # ---------------------------
        # Output dirs
        # ---------------------------
        scene_dir = os.path.join(out_root, base)
        img_dir = os.path.join(scene_dir, "images")
        ann_dir = os.path.join(scene_dir, "annotations")
        seq_dir = os.path.join(scene_dir, "sequences")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(seq_dir, exist_ok=True)

        # --------------------------
        # Build agent lookup table
        # --------------------------
        agent_info = {
            a_token: {
                "type": a.get("type", "unknown"),
                "size": a.get("size", [4.0, 2.0])
            }
            for a_token, a in agents.items()
        }

        # --------------------------
        # Prepare static obstacles
        # --------------------------
        static_objs = []
        xs, ys = [], []

        for oid, ob in obstacles.items():
            p = ob.get("coords") or ob.get("center") or ob.get("position")
            if p is None:
                continue

            x, y = float(p[0]), float(p[1])
            size = ob.get("size", [2.0, 2.0])
            heading = float(ob.get("heading", 0.0))

            static_objs.append({
                "id": oid,
                "type": ob.get("type", "obstacle"),
                "size": size,
                "position": (x, y),
                "heading": heading,
            })
            xs.append(x)
            ys.append(y)

        # --------------------------
        # Calculate World Bounds
        # --------------------------
        # Collect dynamic instance positions for bounds
        for inst in instances.values():
            try:
                x, y = try_get_position(inst)
                xs.append(x)
                ys.append(y)
            except:
                pass

        if not xs:  # Fallback if empty scene
            xs, ys = [0.0], [0.0]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        margin = params["world_margin_m"]
        bbox = (xmin - margin, xmax + margin, ymin - margin, ymax + margin)

        # --------------------------
        # Frame processing
        # --------------------------
        frame_tokens = sorted(frames.keys(), key=lambda t: frames[t]["timestamp"])
        W = H = params["bev_size"]

        for idx, ftoken in enumerate(frame_tokens):
            frame = frames[ftoken]
            timestamp = frame.get("timestamp", 0.0)

            # Collect dynamic objects
            dyn_objs = []

            # Pre-fetch instance data to avoid repeated lookups
            frame_instances = frame.get("instances", [])

            for inst_token in frame_instances:
                inst = instances.get(inst_token)
                if inst is None:
                    continue

                try:
                    x, y = try_get_position(inst)
                except:
                    continue

                agent_token = inst.get("agent_token") or inst.get("agent")
                a_info = agent_info.get(agent_token, {})

                dyn_objs.append({
                    "id": inst_token,
                    "type": a_info.get("type", "unknown"),
                    "size": a_info.get("size", [4.0, 2.0]),
                    "position": (x, y),
                    "heading": try_get_heading(inst),
                    "agent_token": agent_token,
                })

            # --------------------------
            # Create BEV image (Rendering)
            # --------------------------
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

            # Draw static objects
            for s in static_objs:
                pts = get_rotated_corners_vectorized(
                    s["position"][0], s["position"][1], s["size"][0], s["size"][1], s["heading"]
                )
                color = color_for_type(s["type"])
                draw_rotated_box_fast(canvas, pts, bbox, W, H, color, fill=True)

            # Draw dynamic objects
            for o in dyn_objs:
                pts = get_rotated_corners_vectorized(
                    o["position"][0], o["position"][1], o["size"][0], o["size"][1], o["heading"]
                )
                color = color_for_type(o["type"])
                draw_rotated_box_fast(canvas, pts, bbox, W, H, color, fill=True)

            # Save BEV image
            img_name = f"{idx:06d}_{ftoken[:8]}.png"
            # Using cv2.imencode -> tofile is faster on some systems, but imwrite is standard
            cv2.imwrite(os.path.join(img_dir, img_name), canvas)

            # --------------------------
            # Annotation JSON
            # --------------------------
            ann_dyn_objects = []
            for o in dyn_objs:
                pts = get_rotated_corners_vectorized(
                    o["position"][0], o["position"][1], o["size"][0], o["size"][1], o["heading"]
                )
                # Convert numpy points to list of dicts for JSON
                pts_list = [{"x": float(p[0]), "y": float(p[1])} for p in pts]

                ann_dyn_objects.append({
                    "id": o["id"],
                    "agent_token": o["agent_token"],
                    "type": o["type"],
                    "center": {"x": o["position"][0], "y": o["position"][1]},
                    "heading": o["heading"],
                    "size": {"length": o["size"][0], "width": o["size"][1]},
                    "bbox_world": pts_list
                })

            ann = {
                "frame_token": ftoken,
                "timestamp": timestamp,
                "bev_image": img_name,
                "bbox_world": {
                    "xmin": bbox[0], "xmax": bbox[1],
                    "ymin": bbox[2], "ymax": bbox[3],
                },
                "static_objects": static_objs,
                "dynamic_objects": ann_dyn_objects
            }

            with open(os.path.join(ann_dir, img_name.replace(".png", ".json")), "w") as f:
                json.dump(ann, f, indent=2)

        # --------------------------
        # Build sequences (10 sec @ fps)
        # --------------------------
        fps = params["fps"]
        seq_secs = params["seq_secs"]
        seq_len = fps * seq_secs

        # Optimization: Generate sequences in memory first
        sequences_meta = []
        for start in range(0, len(frame_tokens) - seq_len + 1, seq_len):
            seq_id = f"seq_{start:06d}"
            seq_path = os.path.join(seq_dir, seq_id)
            os.makedirs(seq_path, exist_ok=True)

            seq_frames = frame_tokens[start:start + seq_len]
            current_seq = {
                "sequence_id": seq_id,
                "frames": []
            }

            for ftoken in seq_frames:
                idx_global = frame_tokens.index(ftoken)
                img_name = f"{idx_global:06d}_{ftoken[:8]}.png"
                current_seq["frames"].append({
                    "frame_idx": idx_global,
                    "frame_token": ftoken,
                    "image": os.path.join("..", "..", "images", img_name),
                    "annotation": os.path.join("..", "..", "annotations", img_name.replace(".png", ".json"))
                })

            with open(os.path.join(seq_path, "sequence.json"), "w") as f:
                json.dump(current_seq, f, indent=2)
            sequences_meta.append(current_seq)

        return {"scene": base, "status": "ok"}

    except Exception as e:
        return {"scene": base, "status": "error", "error": str(e)}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bev_size", type=int, default=1000)
    parser.add_argument("--world_margin_m", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seq_secs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    groups = find_scene_groups(args.input_dir)
    if not groups:
        print("[ERROR] No *_scene.json found")
        return

    params = {
        "bev_size": args.bev_size,
        "world_margin_m": args.world_margin_m,
        "fps": args.fps,
        "seq_secs": args.seq_secs,
    }

    print(f"[INFO] Found {len(groups)} scenes. Starting generation with {args.workers} workers...")

    # Prepare arguments for multiprocessing
    # Each worker needs: (base, files, out_root, params)
    tasks = []
    for base, files in groups.items():
        tasks.append((base, files, args.output_dir, params))

    summary = []

    # Use ProcessPoolExecutor for true parallelism (bypassing GIL)
    # This is the standard "DLP method" for dataset generation
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_scene, task) for task in tasks]

        # Use tqdm for progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Scenes"):
            result = future.result()
            summary.append(result)
            if result["status"] == "error":
                print(f"\n[ERROR] Scene {result['scene']} failed: {result['error']}")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[ALL DONE]")


if __name__ == "__main__":
    main()