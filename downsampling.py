import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description="Perform downsampling and upsampling on voxel data.")
    parser.add_argument('--category', type=str, required=True, help="Category name (e.g., 'chair').")
    args = parser.parse_args()

    category = args.category
    input_path = f"./data/hdf5_data/{category}_voxels_train.npy"
    maxpool_output_path = f"./data/hdf5_data/{category}_voxels_train_maxpool.npy"
    avgpool_output_path = f"./data/hdf5_data/{category}_voxels_train_avgpool.npy"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file for category '{category}' not found at {input_path}.")
    
    np_data = np.load(input_path)
    tensor = torch.from_numpy(np_data).float()

    maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
    maxpool_tensor = maxpool(tensor)
    maxpool_np = maxpool_tensor.numpy()
    np.save(maxpool_output_path, maxpool_np)

    avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
    avgpool_tensor = avgpool(tensor)
    avgpool_np = avgpool_tensor.numpy()
    np.save(avgpool_output_path, avgpool_np)

    print(f"Processing for category '{category}' completed.")
    print(f"MaxPool output saved to: {maxpool_output_path}")
    print(f"AvgPool output saved to: {avgpool_output_path}")

if __name__ == "__main__":
    main()
