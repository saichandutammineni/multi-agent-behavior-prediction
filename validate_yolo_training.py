import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Configuration (Must match your training script) ---
# Path where your training script saved the best weights
# Note: 'run_03' comes from your training script's 'name' argument
# WEIGHTS_PATH = "/Users/sujkrish/Downloads/best_drive.pt"
WEIGHTS_PATH = "/Users/sujkrish/Desktop/UCI/Fall_2025/271P_AI/project/Parking-Environment-Behaviour-Prediction/yolov8n-obb.pt"
DATA_YAML = "./yolo_dataset/dataset.yaml"
VAL_IMAGES_PATH = "./yolo_dataset/images/val"
IMG_SIZE = 1024


def print_metrics(metrics):
    """Parses and prints the metrics object nicely."""
    print("\n" + "=" * 40)
    print(f"   YOLOv8 OBB EVALUATION RESULTS")
    print("=" * 40)

    # Box metrics (OBB uses box attributes in Ultralytics)
    # Note: For OBB models, 'box' usually refers to the oriented box metrics
    print(f"mAP50 (Mean Average Precision @ IoU 0.5): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (Overall Performance):           {metrics.box.map:.4f}")
    print(f"Precision (Mean):                         {metrics.box.mp:.4f}")
    print(f"Recall (Mean):                            {metrics.box.mr:.4f}")

    print("-" * 40)
    print("CLASS-WISE PERFORMANCE (mAP50):")
    # names maps ID to class name, maps is the list of map50 per class
    for i, ap in enumerate(metrics.box.maps):
        class_name = metrics.names[i]
        # maps array index usually corresponds to class index
        print(f" - {class_name:<12}: {ap:.4f}")
    print("=" * 40 + "\n")


def run_evaluation():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
        print("Did the training finish successfully?")
        return

    print(f"Loading Model from: {WEIGHTS_PATH}")
    # model = YOLO(WEIGHTS_PATH)
    model = YOLO("yolo11l.pt")

    print("Running Validation on Test Set...")
    # val() automatically handles OBB metrics if the model is OBBf
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=8,
        device='mps',
        split='val',
        save_json=False  # Set True if you want a .json report
    )

    print_metrics(metrics)
    return model


def visualize_predictions(model, num_samples=3):
    """
    Runs inference on random validation images and plots them
    to verify the rotation is working visually.
    """
    print(f"--- Visualizing {num_samples} Random Validation Samples ---")

    # Get list of images
    image_files = glob.glob(os.path.join(VAL_IMAGES_PATH, "*.png"))
    if not image_files:
        print("No validation images found to visualize.")
        return

    # Pick random images
    samples = random.sample(image_files, min(len(image_files), num_samples))

    plt.figure(figsize=(15, 5 * num_samples))

    for i, img_path in enumerate(samples):
        # Run inference
        # conf=0.25 is standard visualization threshold
        results = model.predict(
            img_path,
            imgsz=IMG_SIZE,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        # Plot
        # Ultralytics results[0].plot() returns a BGR numpy array
        res_plotted = results[0].plot()

        # Convert BGR to RGB for Matplotlib
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(res_rgb)
        plt.axis('off')
        plt.title(f"Prediction: {os.path.basename(img_path)}")

    plt.tight_layout()
    plt.show()
    print("Visual Check Complete.")


if __name__ == "__main__":
    # 1. Run Quantitative Metrics
    trained_model = run_evaluation()

    # 2. Run Qualitative Visual Check (Only if model loaded)
    if trained_model:
        visualize_predictions(trained_model)