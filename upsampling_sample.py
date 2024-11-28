import torch
import torch.nn as nn  # Import nn module
import numpy as np
import argparse
from tqdm import tqdm

# Decoder model definition (Encoder is not used)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.layers(x)
        return x

# Decoding function
def decode_images(model, encoded_data, device):
    model.eval()  # Set the model to evaluation mode
    decoded_images = []

    # Decode the data in batches
    with torch.no_grad():  # Disable gradient calculation
        for image in encoded_data:
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 32, 32, 32)
            decoded_image = model(image)
            decoded_images.append(decoded_image.squeeze(0).cpu().numpy())  # Convert to (32, 32, 32)

    return np.array(decoded_images)

# Main function
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    category = args.category

    # Default number of points for each category
    default_point = [12500, 4500, 2000]
    if category == 'chair':
        idx = 0
    elif category == 'airplane':
        idx = 1
    elif category == 'table':
        idx = 2
    num_of_points = default_point[idx]

    # Load the pre-trained decoder model
    model = Decoder().to(device)
    state_dict = torch.load(f'./{category}_decoder.ckpt')  # Load decoder's trained parameters from .ckpt

    # Remove "decoder." prefix from keys in the state dict
    state_dict_modified = {key.replace("decoder.", ""): value for key, value in state_dict.items()}

    # Load the modified state dict into the model
    model.load_state_dict(state_dict_modified)

    # Load the pre-encoded data (1000 samples of 32x32x32 images)
    encoded_data = np.load(f'/root/Diffusion-Project-3DVolume/samples/{category}/{category}_combined_data.npy')
    print(f"Encoded data shape: {encoded_data.shape}")  # Should be (1000, 32, 32, 32)

    # Decode the images
    decoded_images = decode_images(model, encoded_data, device)

    # Remove singleton channel dimension
    decoded_images = decoded_images.squeeze(1)

    # Convert numpy array to PyTorch tensor
    voxels_tensor = torch.from_numpy(decoded_images)

    # Initialize tensor to store binary results
    binary_result = torch.zeros_like(voxels_tensor)

    # Process each sample in the tensor
    for i in tqdm(range(voxels_tensor.shape[0])):
        # Flatten the tensor to 1D (for the current sample)
        flattened_data = voxels_tensor[i].view(-1)

        # Find the value of the k-th largest element
        top_k_value = torch.topk(flattened_data, num_of_points).values[-1].item()

        # Set values greater than top_k_value to 1, others to 0
        binary_result[i] = (voxels_tensor[i] > top_k_value).to(dtype=torch.int32)

    # Save the decoded binary images
    np.save(f'{category}_output.npy', binary_result)  # Save the decoded binary images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Category name (e.g., 'chair').")
    args, unknown = parser.parse_known_args()
    main(args)
