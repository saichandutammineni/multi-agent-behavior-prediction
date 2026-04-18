import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# --- CONFIGURATION ---
DATA_PATH_X = "/Users/sujkrish/Downloads/lstm_train_X.pt"  # From generate_training_data.py
DATA_PATH_Y = "/Users/sujkrish/Downloads/lstm_train_Y.pt"
MODEL_SAVE_PATH = "parking_lstm_10122025.pth"

BATCH_SIZE = 16  # Small batch size for stability with sparse data
HIDDEN_SIZE = 64  # Enough capacity for parking lot physics
LEARNING_RATE = 0.001
EPOCHS = 10
VAL_SPLIT = 0.2  # 20% data for validation

# Class IDs for weighting (1=Pedestrian, 0=Car based on your map)
# We penalize errors on Pedestrians 2x more for safety
PEDESTRIAN_CLASS_ID = 1
PED_WEIGHT = 2.0


# --- MODEL DEFINITION ---
class ContextAwareTracker(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, pred_len=4):
        super(ContextAwareTracker, self).__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size

        # Encoder: [dx, dy, class, static_x, static_y]
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Output Head: Maps hidden state -> velocity (dx, dy)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: (Batch, 6, 5)

        # 1. ENCODE
        _, (h, c) = self.encoder(x)

        # 2. DECODE
        outputs = []

        # Get static context from last input frame (Indices 2,3,4)
        # We assume Class + Static Obstacle pos are constant for prediction window
        static_context = x[:, -1, 2:]  # (Batch, 3)

        # Initial Decoder Input: Last observed velocity + Static Context
        last_velocity = x[:, -1, :2]
        decoder_input = torch.cat((last_velocity, static_context), dim=1).unsqueeze(1)

        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))

            # Predict Velocity
            pred_velocity = self.linear(out)  # (Batch, 1, 2)
            outputs.append(pred_velocity)

            # Autoregressive Step: Use PREDICTED velocity for next input
            decoder_input = torch.cat((pred_velocity, static_context.unsqueeze(1)), dim=2)

        return torch.cat(outputs, dim=1)  # (Batch, 4, 2)


# --- UTILS ---
def calculate_metrics(pred_vel, target_vel):
    """
    Calculates Average Displacement Error (ADE) and Final Displacement Error (FDE).
    Since inputs are Velocities, we must CumSum to get positions.
    """
    # CumSum to get positions relative to t=0
    pred_pos = torch.cumsum(pred_vel, dim=1)
    target_pos = torch.cumsum(target_vel, dim=1)

    # Euclidean Error per step
    errors = torch.norm(pred_pos - target_pos, dim=2)  # (Batch, 4)

    ade = torch.mean(errors)  # Mean over time and batch
    fde = torch.mean(errors[:, -1])  # Mean error at final step
    return ade.item(), fde.item()


def weighted_mse_loss(pred, target, inputs):
    """
    Standard MSE, but multiplied by PED_WEIGHT if the object is a Pedestrian.
    """
    # Calculate raw squared error
    loss = (pred - target) ** 2  # Shape: [Batch, 4, 2]

    # Extract class ID from input (Sequence index 0, Feature index 2)
    # Shape: (Batch, 1, 1)
    class_ids = inputs[:, 0, 2].view(-1, 1, 1)

    # Create weight tensor
    weights = torch.ones_like(loss)  # Shape: [Batch, 4, 2]

    # 1. Create Boolean Mask (Batch, 1, 1)
    mask = (class_ids == PEDESTRIAN_CLASS_ID)

    # 2. Expand mask to match Loss dimensions (Batch, 4, 2)
    mask = mask.expand_as(weights)

    # 3. Apply weights
    weights[mask] = PED_WEIGHT

    return torch.mean(loss * weights)


# --- TRAINING LOOP ---
def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH_X):
        print(f"Error: {DATA_PATH_X} not found. Run generate_training_data.py first.")
        return

    X = torch.load(DATA_PATH_X)  # (N, 6, 5)
    Y = torch.load(DATA_PATH_Y)  # (N, 4, 2)

    print(f"Loaded Dataset: {X.shape[0]} samples")

    # 2. Split Data
    dataset = TensorDataset(X, Y)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():  # For completeness (won't be true on Mac)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ContextAwareTracker().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'ade': [], 'fde': []}

    print(f"Starting training on {device}...")

    for epoch in tqdm(range(EPOCHS)):
        # --- TRAIN ---
        model.train()
        train_loss_accum = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            pred_Y = model(batch_X)

            # Loss: Predicted Velocities vs Target Velocities
            loss = weighted_mse_loss(pred_Y, batch_Y, batch_X)

            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        val_loss_accum = 0
        ade_accum, fde_accum = 0, 0

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                pred_Y = model(batch_X)

                loss = weighted_mse_loss(pred_Y, batch_Y, batch_X)
                val_loss_accum += loss.item()

                # Metrics (in Meters)
                ade, fde = calculate_metrics(pred_Y, batch_Y)
                ade_accum += ade
                fde_accum += fde

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_ade = ade_accum / len(val_loader)
        avg_fde = fde_accum / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Log
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ADE: {avg_ade:.2f}m")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nTraining Complete. Best Model saved to {MODEL_SAVE_PATH}")
    print(f"Final Performance -> ADE: {avg_ade:.2f}m (Avg Error), FDE: {avg_fde:.2f}m (Final Error)")

    # --- PLOT ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig('training_curve.png')
    print("Saved training_curve.png")


if __name__ == "__main__":
    main()