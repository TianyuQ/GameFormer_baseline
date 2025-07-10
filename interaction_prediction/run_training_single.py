#!/usr/bin/env python3
"""
Simple script to run single-process training for GameFormer interaction prediction.
This script demonstrates how to run the training without distributed training.
"""

import subprocess
import sys
import os

def run_training():
    """
    Run the training script with single process configuration.
    """
    
    # Training parameters
    cmd = [
        sys.executable, "train.py",
        "--batch_size", "8",  # Reduced batch size for single GPU
        "--training_epochs", "30",
        "--learning_rate", "1e-4",
        "--seed", "3407",
        "--device", "cuda",  # Use CUDA if available, otherwise CPU
        "--name", "SingleProcess_Exp",
        "--train_set", "path/to/train/data",  # Replace with actual path
        "--valid_set", "path/to/valid/data",  # Replace with actual path
        "--level", "3",
        "--neighbors_to_predict", "1",
        "--modalities", "6",
        "--future_len", "80",
        "--encoder_layers", "6"
    ]
    
    print("Running single-process training with command:")
    print(" ".join(cmd))
    print("\nNote: Make sure to replace the data paths with actual paths to your processed data.")
    print("The processed data should be .npz files created by the data_process.py script.")
    
    # Uncomment the following lines to actually run the training:
    # try:
    #     subprocess.run(cmd, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Training failed with error: {e}")
    # except FileNotFoundError:
    #     print("Error: train.py not found. Make sure you're in the correct directory.")

def check_requirements():
    """
    Check if required packages are available.
    """
    required_packages = ['torch', 'numpy', 'matplotlib', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {missing_packages}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All required packages are available.")
    return True

def main():
    print("GameFormer Single-Process Training Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    print("\nKey Changes Made to train.py:")
    print("1. Removed distributed training (DDP, DistributedSampler)")
    print("2. Removed multiprocessing (num_workers=0)")
    print("3. Added device parameter (cuda/cpu)")
    print("4. Simplified logging (no rank checks)")
    print("5. Removed distributed process group initialization")
    
    print("\nTo run training:")
    print("1. First process your data using data_process.py")
    print("2. Update the train_set and valid_set paths in this script")
    print("3. Run: python run_training_single.py")
    
    # Ask user if they want to run training
    response = input("\nDo you want to run the training now? (y/n): ")
    if response.lower() == 'y':
        run_training()

if __name__ == "__main__":
    main() 