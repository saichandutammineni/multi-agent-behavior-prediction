import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from data_new import ParkingSequenceLoader

# --- CONFIGURATION ---
DATASET_ROOT = "./bev-dataset-mini"  # Update this to your dataset path
SAVE_PATH_X = "lstm_train_X.pt"
SAVE_PATH_Y = "lstm_train_Y.pt"
MODEL_PATH = '/Users/sujkrish/Downloads/best.pt'  # Path to your fine-tuned OBB model

# Physics Params
PIXELS_PER_METER = 15.0  # <--- CRITICAL: Set this based on your map scale
HISTORY_LEN = 6  # 6 seconds input
PREDICTION_LEN = 4  # 4 seconds output

# CRITICAL: Check your model.names to set this map correctly!
# If your fine-tuned model has 0=car, 1=pedestrian, use: {0: 1.0, 1: 0.0}
# If using standard DOTA-trained weights, 'small-vehicle' is usually 9, 'large-vehicle' is 10.
CLASS_MAP = {0: 0, 1: 1}

# Load the OBB Model
yolo_model = YOLO(MODEL_PATH)


def is_static_object(history):
    """
    Determines if an object is static based on total displacement.
    history: List of (x, y) tuples.
    """
    if len(history) < 2: return False
    start = np.array(history[0])
    end = np.array(history[-1])
    # If moved less than 0.5m in 6 seconds, it's static
    return np.linalg.norm(end - start) < 0.5


def get_nearest_static_interaction(curr_pos, static_positions):
    """
    Finds the relative vector (dx, dy) to the nearest static object.
    """
    if not static_positions:
        return [100.0, 100.0]  # Return large distance if no static objects

    curr_pos = np.array(curr_pos)
    statics = np.array(static_positions)

    # Calculate Euclidean distance to all static objects
    dists = np.linalg.norm(statics - curr_pos, axis=1)
    min_idx = np.argmin(dists)

    # Return vector: Static_Pos - Current_Pos
    return (statics[min_idx] - curr_pos).tolist()


def process_sequence(sequence):
    """
    Runs OBB tracking on a sequence and generates LSTM training tensors.
    """
    track_history = {}  # {id: [(x,y), ...]}
    track_classes = {}  # {id: class_mapped_value}

    # --- PASS 1: TRACKING (OBB SPECIFIC) ---
    for frame_obj in sequence:
        img = frame_obj.load_image()

        # Run OBB Tracking
        # persist=True allows ID tracking across frames
        results = yolo_model.track(img, persist=True, verbose=False)

        # OBB check: Ensure we have OBB detections and IDs
        if results[0].obb is None or results[0].obb.id is None:
            continue

        # Extract OBB Data: xywhr (x, y, w, h, rotation)
        obb_data = results[0].obb.xywhr.cpu().numpy()
        ids = results[0].obb.id.int().cpu().numpy()
        clss = results[0].obb.cls.int().cpu().numpy()

        for obb, t_id, cls in zip(obb_data, ids, clss):
            if cls not in CLASS_MAP:
                continue

            # OBB Format is [cx, cy, w, h, rotation]
            # We take cx (0) and cy (1) and convert to meters
            cx, cy = obb[0] / PIXELS_PER_METER, obb[1] / PIXELS_PER_METER

            if t_id not in track_history:
                track_history[t_id] = []

            track_history[t_id].append((cx, cy))
            track_classes[t_id] = CLASS_MAP[cls]

    # --- PASS 2: IDENTIFY STATIC OBJECTS ---
    # We must identify what is static *globally* for this sequence
    static_ids = set()
    for t_id, positions in track_history.items():
        # If object existed long enough and didn't move much -> Static
        if len(positions) >= HISTORY_LEN and is_static_object(positions):
            static_ids.add(t_id)

    # --- PASS 3: GENERATE TRAINING SAMPLES ---
    X_samples = []
    Y_samples = []

    for t_id in track_history:
        # We do not predict trajectories for static objects
        if t_id in static_ids: continue

        positions = track_history[t_id]

        # Check if track is long enough for (History + Prediction)
        if len(positions) < (HISTORY_LEN + PREDICTION_LEN): continue

        # Sliding Window Generation
        # We slide through the object's life to generate multiple training examples
        for i in range(len(positions) - HISTORY_LEN - PREDICTION_LEN + 1):
            # 1. Input History (6 frames)
            hist_pos = np.array(positions[i: i + HISTORY_LEN])

            # 2. Target Future (4 frames)
            fut_pos = np.array(positions[i + HISTORY_LEN: i + HISTORY_LEN + PREDICTION_LEN])

            # 3. Calculate Input Velocities (dx, dy)
            # np.diff reduces size by 1, so we pad the first velocity
            vel_in = np.diff(hist_pos, axis=0)
            vel_in = np.vstack([vel_in[0], vel_in])  # Shape: (6, 2)

            # 4. Calculate Target Velocities (dx, dy)
            # Predict velocity relative to the LAST history frame
            full_segment = np.concatenate([hist_pos[-1:], fut_pos], axis=0)
            vel_out = np.diff(full_segment, axis=0)  # Shape: (4, 2)

            # 5. Calculate Interaction Feature (Nearest Static Object)
            # We use the position at the END of the history window (t=0)
            curr_pos = hist_pos[-1]

            # Get positions of all static objects (using their last known pos)
            static_positions = [track_history[sid][-1] for sid in static_ids]

            interaction_vec = get_nearest_static_interaction(curr_pos, static_positions)

            # Repeat interaction vector for all 6 input steps
            # (Assuming interaction is constant relative to "now")
            inter_feat = np.tile(interaction_vec, (HISTORY_LEN, 1))  # (6, 2)

            # 6. Class Feature
            c_val = track_classes[t_id]
            class_feat = np.full((HISTORY_LEN, 1), c_val)  # (6, 1)

            # 7. Concatenate all features into input Tensor X
            # [vel_x, vel_y, class, inter_x, inter_y]
            x_tensor = np.concatenate([vel_in, class_feat, inter_feat], axis=1)  # (6, 5)

            X_samples.append(x_tensor)
            Y_samples.append(vel_out)

    return X_samples, Y_samples


def main():
    # Initialize Loader
    print(f"Loading data from {DATASET_ROOT}...")
    dataset = ParkingSequenceLoader(DATASET_ROOT)

    all_X = []
    all_Y = []

    print(f"Processing sequences with model: {MODEL_PATH}...")

    # Iterate over sequences
    for seq in tqdm(dataset):
        # Reset tracker state between videos so IDs start fresh
        if hasattr(yolo_model.predictor, 'trackers'):
            if yolo_model.predictor.trackers:
                yolo_model.predictor.trackers[0].reset()

        x, y = process_sequence(seq)
        all_X.extend(x)
        all_Y.extend(y)

    if not all_X:
        print("Error: No valid trajectories generated.")
        print("Check: 1. Is CLASS_MAP correct? 2. Is PIXELS_PER_METER reasonable? 3. Are images loading?")
        print(f"Model detected classes: {yolo_model.names}")
        return

    # Convert to PyTorch Tensors
    final_X = torch.tensor(np.array(all_X), dtype=torch.float32)
    final_Y = torch.tensor(np.array(all_Y), dtype=torch.float32)

    print(f"\n--- SUCCESS ---")
    print(f"Input Shape (X): {final_X.shape} -> (Samples, 6, 5)")
    print(f"Target Shape (Y): {final_Y.shape} -> (Samples, 4, 2)")

    torch.save(final_X, SAVE_PATH_X)
    torch.save(final_Y, SAVE_PATH_Y)
    print(f"Saved to {SAVE_PATH_X} and {SAVE_PATH_Y}")


if __name__ == "__main__":
    main()