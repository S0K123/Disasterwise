"""
Sample execution script for the Disasterwise project.
This script demonstrates how to run each module of the research-grade pipeline.
"""

import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        # We use shell=True on Windows if needed, but for simple python calls it's fine
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        # sys.exit(1) # Uncomment to stop on error

def main():
    # 1. Preprocessing
    run_command(
        [sys.executable, "main.py", "preprocess", "--test_size", "0.2", "--val_size", "0.1"],
        "Data Preprocessing and Splitting"
    )

    # 2. Tiling (New Step)
    run_command(
        [sys.executable, "main.py", "tile", "--tile_size", "512", "--overlap", "50"],
        "Satellite Image Tiling"
    )

    # 3. Train Detection (Example with mock or pre-trained weights)
    # Note: This requires the dataset to be in the correct format
    # run_command(
    #     [sys.executable, "main.py", "train-detection", "--model", "rtdetr-l.pt", "--epochs", "10"],
    #     "RT-DETR Detection Training"
    # )

    # 3. Train Prediction
    # Note: Requires features to be extracted first
    # run_command(
    #     [sys.executable, "main.py", "train-prediction", "--epochs", "20", "--input_dim", "20"],
    #     "Transformer Risk Prediction Training"
    # )

    # 4. End-to-End Inference
    # Note: Requires trained model weights
    print("\n[INFO] End-to-End Inference example:")
    print("python main.py inference --image data/raw/xview/train/images/socal-fire_00000001_post_disaster.png --det_model saved_models/rtdetr_best.pt --pred_model saved_models/transformer_best.pt")

    # 5. Explainability (XAI)
    # run_command(
    #     [sys.executable, "main.py", "explain", "--model", "saved_models/transformer_best.pt", "--data", "data/processed/val_features.npy"],
    #     "Explainable AI Analysis"
    # )

    print("\nPipeline overview complete. Please ensure model weights and data are present before running the full training/inference.")

if __name__ == "__main__":
    main()
