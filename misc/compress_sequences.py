import json
import os
from pathlib import Path

def sample_sequence_file(input_path, output_path, target_frames=10):
    """
    Reads a sequence.json file, reduces frames to target count, and saves to output_path.
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        total_frames = len(frames)
        
        if total_frames == 0:
            print(f"    [!] Skipping {input_path.parent.name}: No frames found.")
            return

        # Calculate step size. e.g. 250 / 10 = 25.
        # Use max(1, ...) to avoid division by zero or step of 0 if total_frames < target
        step = max(1, int(total_frames / target_frames))
        
        # Slice the list [start:stop:step]
        data['frames'] = frames[::step]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"    [✓] {input_path.parent.name}: {total_frames} -> {len(data['frames'])} frames saved to {output_path.name}")
        
    except Exception as e:
        print(f"    [X] Error processing {input_path}: {e}")

def process_dataset(root_path):
    base_dir = Path(root_path).expanduser() # Handle the ~ in the path
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    # 1. Find all Scene directories (DJI_00XX)
    # We sort them so the output log is readable
    scene_dirs = sorted([d for d in base_dir.glob("DJI_*") if d.is_dir()])
    
    if not scene_dirs:
        print(f"No 'DJI_XXXX' directories found in {base_dir}")
        return

    print(f"Found {len(scene_dirs)} scenes to process...")

    for scene_dir in scene_dirs:
        print(f"\nProcessing Scene: {scene_dir.name}")
        
        # 2. Go into /sequences
        sequences_path = scene_dir / "sequences"
        
        if not sequences_path.exists():
            print(f"  [!] No 'sequences' folder found in {scene_dir.name}")
            continue
            
        # 3. Find all sequence directories (seq_XXXXXX)
        # Using glob("seq_*") automatically filters out "backup_XXXXXX"
        seq_dirs = sorted([d for d in sequences_path.glob("seq_*") if d.is_dir()])
        
        for seq_dir in seq_dirs:
            input_json = seq_dir / "sequence.json"
            output_json = seq_dir / "sampled_sequence.json"
            
            if input_json.exists():
                sample_sequence_file(input_json, output_json)

if __name__ == "__main__":
    # The root path where your DJI_XXXX folders are located
    ROOT_PATH = "~/De/U/Fall_2025/271P_AI/project/dlp-dataset/dlp/bev-dataset"
    
    process_dataset(ROOT_PATH)
