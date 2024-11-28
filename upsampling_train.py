import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm  # tqdm for progress bar

# Encoder: Downsamples the input using Max Pooling
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)  # Downsample (1, number of data, 64, 64, 64) -> (1, number of data, 32, 32, 32)

    def forward(self, x):
        return self.pool(x)

# Decoder: Upsamples the input and restores the original dimensions
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # (32 -> 64)

        self.layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),  # Channel 1 -> 16
            nn.ReLU(),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),  # Channel 16 -> 32
            nn.ReLU(),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),  # Channel 32 -> 64
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 1, kernel_size=3, padding=1)  # Channel 64 -> 1 (final output)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.layers(x)
        return x

# Full Network: Combines Encoder and Decoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training Loop with tqdm for epochs
def train(model, data, criterion, optimizer, num_epochs, batch_size, device):
    model.train()
    num_samples = data.shape[0]  # Number of data points
    
    # tqdm applied to the number of epochs
    with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as epoch_pbar:
        for epoch in range(num_epochs):
            epoch_loss = 0

            # Randomly sample a subset of the data for each epoch
            indices = np.random.choice(num_samples, batch_size, replace=False)  # Randomly pick indices
            batch_data = data[indices]  # Select corresponding data

            for idx in range(batch_size):
                image = batch_data[idx]  # Select one image (64, 64, 64)
                image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dimensions

                # Forward pass
                reconstructed = model(image)
                loss = criterion(reconstructed, image)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Update tqdm bar for the epoch and display the loss
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({"Loss": epoch_loss / batch_size})  # Display average loss for the epoch

# Main function
def main(args):
    category = args.category
    input_path = f"./data/hdf5_data/{category}_voxels_train.npy"
    learning_rate = 1e-3
    num_epochs = 400
    batch_size = 20  # Set batch size for each epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data directly from the file
    data = np.load(input_path)  # Load the .npy file directly
    data = data.astype(np.float32)  # Ensure correct data type
    
    # Model, Loss, Optimizer
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for image reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, data, criterion, optimizer, num_epochs, batch_size, device)

    # Save the trained model with .ckpt extension
    model_path = f'{category}_decoder.ckpt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Category name (e.g., 'chair').")
    args, unknown = parser.parse_known_args()
    main(args)
