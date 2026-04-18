import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from ultralytics import YOLO
from data_loader import ParkingSequenceLoader

# --- CONFIGURATION ---
DATASET_ROOT = "./data/parking_lot"
MODEL_PATH_YOLO = "yolov8n-obb.pt"
MODEL_PATH_LSTM = "parking_lstm_best.pth"
OUTPUT_FILE = "sequence_plot.png"

PIXELS_PER_METER = 20.0
HISTORY_LEN = 6
PRED_LEN = 4
CLASS_MAP = {0: 1.0, 1: 0.0}  # 1.0=Car, 0.0=Pedestrian
CLASS_LABELS = {1.0: "Car", 0.0: "Pedestrian"}


# --- MODEL DEFINITION ---
class ContextAwareTracker(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=64, pred_len=4):
        super(ContextAwareTracker, self).__init__()
        self.pred_len = pred_len
        self.encoder = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        outputs = []
        static_context = x[:, -1, 2:]
        last_velocity = x[:, -1, :2]
        decoder_input = torch.cat((last_velocity, static_context), dim=1).unsqueeze(1)
        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred_velocity = self.linear(out)
            outputs.append(pred_velocity)
            decoder_input = torch.cat((pred_velocity, static_context.unsqueeze(1)), dim=2)
        return torch.cat(outputs, dim=1)


# --- HELPER FUNCTIONS ---
def is_static(history):
    if len(history) < 2: return False
    start, end = np.array(history[0]), np.array(history[-1])
    return np.linalg.norm(end - start) < 0.5


def get_nearest_static_interaction(curr_pos, static_objects):
    if not static_objects: return [100.0, 100.0]
    curr_pos = np.array(curr_pos)
    statics = np.array(static_objects)
    dists = np.linalg.norm(statics - curr_pos, axis=1)
    min_idx = np.argmin(dists)
    return (statics[min_idx] - curr_pos).tolist()


def get_data_for_sequence(seq, yolo, lstm, device):
    """
    Runs tracking and prediction, returning trajectories in METERS.
    Returns:
        trajectories: List of dicts {'id', 'class', 'history': [(x,y)...], 'pred': [(x,y)...]}
    """
    track_history = {}
    track_classes = {}
    static_ids = set()

    # We process the whole sequence to build full histories
    # Then we take the prediction from the *last possible moment* for each object

    for frame_obj in seq:
        img = frame_obj.load_image()
        results = yolo.track(img, persist=True, verbose=False)

        if results[0].obb is None or results[0].obb.id is None: continue

        obb_data = results[0].obb.xywhr.cpu().numpy()
        ids = results[0].obb.id.int().cpu().numpy()
        clss = results[0].obb.cls.int().cpu().numpy()

        for obb, t_id, cls in zip(obb_data, ids, clss):
            if cls not in CLASS_MAP: continue

            # Convert to Meters
            cx_m, cy_m = obb[0] / PIXELS_PER_METER, obb[1] / PIXELS_PER_METER

            if t_id not in track_history: track_history[t_id] = []
            track_history[t_id].append((cx_m, cy_m))
            track_classes[t_id] = CLASS_MAP[cls]

    # Identify Statics (needed for model calculation, even if not shown)
    for t_id, hist in track_history.items():
        if len(hist) >= HISTORY_LEN and is_static(hist):
            static_ids.add(t_id)

    # Prepare Data for Plotting
    final_trajectories = []

    # For every object, we predict based on its *latest* valid window
    for t_id, hist in track_history.items():
        if t_id in static_ids: continue  # Skip statics
        if len(hist) < HISTORY_LEN: continue

        # Take the last 6 steps as input
        input_hist = np.array(hist[-HISTORY_LEN:])
        curr_pos = input_hist[-1]

        # Calculate Context (Interaction)
        # We use the LAST known positions of static objects
        static_locs = [track_history[s][-1] for s in static_ids]
        inter_vec = get_nearest_static_interaction(curr_pos, static_locs)

        # Build Tensor
        vel_in = np.diff(input_hist, axis=0)
        vel_in = np.vstack([vel_in[0], vel_in])
        c_val = track_classes[t_id]
        feat = np.concatenate([
            vel_in,
            np.full((HISTORY_LEN, 1), c_val),
            np.tile(inter_vec, (HISTORY_LEN, 1))
        ], axis=1)

        # Run LSTM
        input_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_vels = lstm(input_tensor).cpu().numpy()[0]  # (4, 2)

        # Reconstruct Prediction Path
        pred_path = []
        cx, cy = curr_pos
        for vx, vy in pred_vels:
            cx += vx
            cy += vy
            pred_path.append((cx, cy))

        final_trajectories.append({
            'id': t_id,
            'class': c_val,
            'history': input_hist,  # Last 6 steps
            'pred': np.array(pred_path)  # Future 4 steps
        })

    return final_trajectories


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    print("Loading Models...")
    yolo = YOLO(MODEL_PATH_YOLO)
    lstm = ContextAwareTracker().to(device)
    lstm.load_state_dict(torch.load(MODEL_PATH_LSTM, map_location=device))
    lstm.eval()

    # 2. Load Sequence
    loader = ParkingSequenceLoader(DATASET_ROOT)
    if len(loader) == 0:
        print("No sequences found.")
        return

    # Pick the first one (or specific index if you want)
    seq = loader[0]
    seq_id = seq.seq_id if hasattr(seq, 'seq_id') else "0"
    print(f"Processing Sequence ID: {seq_id} ({len(seq)} frames)...")

    # 3. Get Trajectories
    # Reset tracker first
    if hasattr(yolo.predictor, 'trackers') and yolo.predictor.trackers:
        yolo.predictor.trackers[0].reset()

    data = get_data_for_sequence(seq, yolo, lstm, device)

    if not data:
        print("No dynamic objects found in this sequence.")
        return

    # 4. Plot (Matplotlib)
    print(f"Plotting {len(data)} objects...")
    plt.figure(figsize=(10, 10))

    for obj in data:
        hist = obj['history']
        pred = obj['pred']
        cls_name = CLASS_LABELS.get(obj['class'], "Unknown")

        # Plot History (Blue Solid)
        plt.plot(hist[:, 0], hist[:, 1], color='blue', linewidth=2, alpha=0.7)
        plt.plot(hist[0, 0], hist[0, 1], 'bo', markersize=4)  # Start dot

        # Plot Prediction (Red Dashed)
        # Connect last history to first pred
        transition_x = [hist[-1, 0], pred[0, 0]]
        transition_y = [hist[-1, 1], pred[0, 1]]
        plt.plot(transition_x, transition_y, color='red', linestyle='--', linewidth=2)
        plt.plot(pred[:, 0], pred[:, 1], color='red', linestyle='--', linewidth=2)

        # Label ID
        mid_x, mid_y = hist[-1]
        plt.text(mid_x, mid_y, f"{cls_name} {obj['id']}", fontsize=9)

    # Formatting
    plt.title(f"Trajectory Prediction - Sequence {seq_id}\n(Blue: Past 6s, Red: Pred 4s)", fontsize=14)
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.axis('equal')  # Keep aspect ratio (so circles look like circles)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='History'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Prediction')
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved plot to {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()