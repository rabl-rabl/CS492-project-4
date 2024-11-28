import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
import math
from typing import Optional
import argparse
from train import UNet3D, DDPMScheduler, DiffusionModule3D

parser = argparse.ArgumentParser(description="3D Diffusion Model Sampling")
parser.add_argument('--category', type=str, required=True, help='Category to generate samples for (e.g., chair, table)')
args = parser.parse_args()
category = args.category

checkpoint_path = f'/root/Diffusion-Project-3DVolume/results/diffusion-voxel-11-24-121026/{category}.ckpt' # Change the folder name

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found for category {category} at {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint['config']

num_train_timesteps = config['num_diffusion_train_timesteps']
beta_1 = config['beta_1']
beta_T = config['beta_T']
image_resolution = config['image_resolution']

network = UNet3D(
    T=num_train_timesteps,
    image_resolution=image_resolution,
    ch=64,
    ch_mult=[1, 2, 4, 8],
    num_res_blocks=2,
    dropout=0.1,
)

var_scheduler = DDPMScheduler(
    num_train_timesteps=num_train_timesteps,
    beta_1=beta_1,
    beta_T=beta_T,
    mode='linear',
)

ddpm = DiffusionModule3D(network, var_scheduler)
ddpm.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ddpm.to(device)
ddpm.eval()

batch_size = 50
num_files = 20

# 결과 저장 경로 설정
output_dir = f'./samples/{category}'
os.makedirs(output_dir, exist_ok=True)

for i in range(num_files):
    samples = ddpm.sample(batch_size=batch_size, return_traj=False)

    samples = samples.squeeze().cpu().numpy()
    np.save(f"{output_dir}/sample_{i}.npy", samples)

# Visualization function with interactive rotation enabled
def visualize_voxel_prob(voxel_grid, prob=0.01):
    flattened_data = voxel_grid.flatten()
    sorted_data = np.sort(flattened_data)

    index = int((1-prob) * len(sorted_data))
    threshold = sorted_data[index]

    # Apply thresholding and reshape to match original dimensions
    voxel_grid = (voxel_grid > threshold).astype(int)

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    occupied_voxels = np.argwhere(voxel_grid == 1)
    ax.scatter(occupied_voxels[:, 0], occupied_voxels[:, 2], occupied_voxels[:, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set the limits for the axes
    ax.set_xlim([0, voxel_grid.shape[0]])
    ax.set_ylim([0, voxel_grid.shape[1]])
    ax.set_zlim([0, voxel_grid.shape[2]])
    
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Move the buffer cursor to the beginning
    plt.close()
    # Convert the buffer into a Pillow Image
    img = Image.open(buf)
    return img

# Visualize generated samples with interactive rotation
num_visualize = 10
for i in range(num_visualize):
    sample = samples[i]
    img = visualize_voxel_prob(sample)
    file_path = f"{output_dir}/image_{i}.png"
    img.save(file_path)

data_dir = output_dir

file_pattern = 'sample_{}.npy'

arrays_list = []

for i in range(num_files):
    file_name = file_pattern.format(i)
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        print(f'Loading {file_name}...')
        array = np.load(file_path)
        if array.shape == (50, 32, 32, 32):
            arrays_list.append(array)

if len(arrays_list) == num_files:
    combined_array = np.concatenate(arrays_list, axis=0)

output_file = f'{category}_combined_data.npy'
output_path = os.path.join(data_dir, output_file)
np.save(output_path, combined_array)
