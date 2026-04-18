import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# Ensure these paths match where your files are located
DATA_PATH_X = "/Users/sujkrish/Downloads/lstm_train_X.pt"
DATA_PATH_Y = "/Users/sujkrish/Downloads/lstm_train_Y.pt"
MODEL_PATH = "parking_lstm_10122025.pth"
OUTPUT_IMAGE = "inference_viz.png"

# Visualization Settings
SAMPLES_TO_PLOT = 6  # Total subplots
COLS = 3
ROWS = 2


# --- MODEL DEFINITION (Must match training script) ---
class ContextAwareTracker(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, pred_len=4):
        super(ContextAwareTracker, self).__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)

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


# --- RECONSTRUCTION UTILS ---
def reconstruct_paths(history_vel, gt_vel, pred_vel):
    """
    Reconstructs absolute positions from velocities for plotting.
    Assumes the agent is at (0,0) at the last history step (t=0).
    """
    # 1. History Path
    # Cumulative sum of velocities
    # We essentially integrate forward, then shift so the LAST point is at (0,0)
    # This aligns the "present" moment to the origin.
    hist_path = torch.cumsum(history_vel, dim=0)
    hist_path = hist_path - hist_path[-1]

    # 2. Future Paths (Ground Truth & Prediction)
    # These start from (0,0), so we just cumsum
    gt_path = torch.cumsum(gt_vel, dim=0)
    pred_path = torch.cumsum(pred_vel, dim=0)

    return hist_path, gt_path, pred_path


def get_class_name(class_id):
    return "Pedestrian" if class_id == 1 else "Car"


# --- MAIN INFERENCE SCRIPT ---
def main():
    # 1. Load Data and Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    device = torch.device("cpu")  # CPU is sufficient for inference on small batches

    print("Loading data...")
    X = torch.load(DATA_PATH_X, map_location=device)  # (N, 6, 5)
    Y = torch.load(DATA_PATH_Y, map_location=device)  # (N, 4, 2)

    print("Loading model...")
    model = ContextAwareTracker()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 2. Select Random Samples
    total_samples = X.shape[0]
    indices = np.random.choice(total_samples, SAMPLES_TO_PLOT, replace=False)
    # Alternatively, specify indices manually if you want to reproduce specific cases:
    # indices = [2, 6, 7, 14, 3, 4]

    # 3. Setup Plot
    fig, axs = plt.subplots(ROWS, COLS, figsize=(18, 12))
    axs = axs.flatten()

    print(f"Running inference on samples: {indices}")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Extract single sample (Batch size = 1)
            input_tensor = X[idx].unsqueeze(0).to(device)  # (1, 6, 5)
            gt_tensor = Y[idx].to(device)  # (4, 2)

            # Run Model
            pred_tensor = model(input_tensor)  # (1, 4, 2)

            # Unpack Data
            # History: indices 0,1 are dx, dy
            hist_vel = input_tensor[0, :, :2].cpu()
            gt_vel = gt_tensor.cpu()
            pred_vel = pred_tensor[0].cpu()

            # Context info (from last frame of input)
            class_id = int(input_tensor[0, -1, 2].item())
            obs_x = input_tensor[0, -1, 3].item()
            obs_y = input_tensor[0, -1, 4].item()

            # Reconstruct Trajectories (Positions)
            path_hist, path_gt, path_pred = reconstruct_paths(hist_vel, gt_vel, pred_vel)

            # --- PLOTTING ---
            ax = axs[i]

            # Plot History (Blue, solid)
            # We append (0,0) to history so it connects visually to the future lines
            hist_x = torch.cat((path_hist[:, 0], torch.tensor([0.0])))
            hist_y = torch.cat((path_hist[:, 1], torch.tensor([0.0])))
            ax.plot(hist_x, hist_y, 'b-', linewidth=2, alpha=0.7, label='History (6s)')
            ax.scatter(hist_x[0], hist_y[0], color='blue', s=30)  # Start point

            # Plot Ground Truth (Green, solid)
            # Prepend (0,0) for visual connectivity
            gt_x = torch.cat((torch.tensor([0.0]), path_gt[:, 0]))
            gt_y = torch.cat((torch.tensor([0.0]), path_gt[:, 1]))
            ax.plot(gt_x, gt_y, 'g-', linewidth=2, alpha=0.7, label='Ground Truth (4s)')

            # Plot Prediction (Red, dashed)
            pred_x = torch.cat((torch.tensor([0.0]), path_pred[:, 0]))
            pred_y = torch.cat((torch.tensor([0.0]), path_pred[:, 1]))
            ax.plot(pred_x, pred_y, 'r.--', linewidth=2, label='Prediction')

            # # Plot Obstacle (Black X)
            # # The obstacle position is relative to the agent at t=0 (which is 0,0)
            # ax.scatter(obs_x, obs_y, color='black', marker='x', s=100, linewidth=3, label='Obstacle')
            #
            # # Connect obstacle to agent with a faint line (like in the image)
            # ax.plot([0, obs_x], [0, obs_y], 'k:', alpha=0.3)

            # Styling
            class_name = get_class_name(class_id)
            ax.set_title(f"Sample {idx}: {class_name}")
            ax.grid(True, linestyle=':', alpha=0.6)

            # Only add legend to the first plot to avoid clutter
            if i == 0:
                ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Visualization saved to {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    main()