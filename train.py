import os
from pathlib import Path
from datetime import datetime
import math
import argparse
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from io import BytesIO
from PIL import Image

import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotmap import DotMap
from pytorch_lightning import seed_everything

matplotlib.use("Agg")

def visualize_voxel_prob(voxel_grid, prob=0.05):
    flattened_data = voxel_grid.flatten()
    sorted_data = np.sort(flattened_data)

    # Calculate the threshold to keep only 5% of the highest intensity voxels
    index = int((1-prob) * len(sorted_data))  # Index for 5% of the highest values
    threshold = sorted_data[index]

    # Apply thresholding and reshape to match original dimensions
    voxel_grid = (voxel_grid > threshold).astype(int)

    # Visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    occupied_voxels = np.argwhere(voxel_grid == 1)
    ax.scatter(occupied_voxels[:, 0], occupied_voxels[:, 2], occupied_voxels[:, 1])
    
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Move the buffer cursor to the beginning
    plt.close()
    # Convert the buffer into a Pillow Image
    img = Image.open(buf)
    return img


# Define VoxelDataset class
class VoxelDataset(Dataset):
    def __init__(self, voxel_file, transform=None, subset_indices=None):

        self.voxels = np.load(voxel_file)  # Shape: (N, D, H, W)
        if subset_indices is not None:
            self.voxels = self.voxels[subset_indices]
        self.transform = transform

    def __len__(self):
        return self.voxels.shape[0]

    def __getitem__(self, idx):
        voxel = self.voxels[idx]  # Shape: (D, H, W)
        voxel = np.expand_dims(voxel, axis=0)  # Shape: (1, D, H, W)
        voxel = torch.from_numpy(voxel).float()  # Change to Float Tensor
        if self.transform:
            voxel = self.transform(voxel)
        return voxel
    
# Define VoxelDataModule class
class VoxelDataModule:
    def __init__(
        self,
        voxel_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        subset_size: Optional[int] = None,
        transform=None,
    ):
        self.voxel_file = voxel_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.subset_size = subset_size

    def setup(self):
        total_voxels = np.load(self.voxel_file).shape[0]
        if self.subset_size is not None:
            subset_size = min(self.subset_size, total_voxels)
            subset_indices = np.random.choice(total_voxels, subset_size, replace=False)
        else:
            subset_indices = None
        self.train_ds = VoxelDataset(
            self.voxel_file,
            transform=self.transform,
            subset_indices=subset_indices
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

def get_timestep_embedding(timesteps, embedding_dim):

    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, dtype=torch.long, device='cuda')

    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        0, half_dim, dtype=torch.float32, device=timesteps.device
    ) / half_dim
    emb = torch.exp(exponent)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

# Define CustomSequential class
class CustomSequential(nn.Sequential):
    def forward(self, input, t_emb=None):
        for module in self._modules.values():
            if isinstance(module, ResBlock3D):
                input = module(input, t_emb)
            elif isinstance(module, AttentionBlock3D):
                input = module(input)
            else:
                input = module(input)
        return input

# Define UNet3D class
class UNet3D(nn.Module):
    def __init__(
        self,
        T: int,
        image_resolution: int,
        ch: int = 64,
        ch_mult: list = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attn: list = [],
        dropout: float = 0.1,
        use_cfg: bool = False,
        cfg_dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.T = T
        self.image_resolution = image_resolution
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        self.num_classes = num_classes

        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        time_embed_dim = ch * 4

        self.time_embed = nn.Sequential(
            nn.Linear(ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(nn.Conv3d(1, ch, kernel_size=3, padding=1))

        input_block_chans = [ch]
        input_ch = ch

        for level, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                out_ch = ch * mult
                layers = [
                    ResBlock3D(input_ch, out_ch, time_embed_dim, dropout)
                ]
                if level in attn:
                    layers.append(AttentionBlock3D(out_ch))
                self.input_blocks.append(CustomSequential(*layers))
                input_ch = out_ch
                input_block_chans.append(input_ch)
            if level != len(ch_mult) - 1:
                self.input_blocks.append(Downsample3D(input_ch))
                input_block_chans.append(input_ch)

        self.middle_block = CustomSequential(
            ResBlock3D(input_ch, input_ch, time_embed_dim, dropout),
            ResBlock3D(input_ch, input_ch, time_embed_dim, dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                layers = [
                    ResBlock3D(input_ch + skip_ch, ch * mult, time_embed_dim, dropout)
                ]
                if level in attn:
                    layers.append(AttentionBlock3D(ch * mult))
                self.output_blocks.append(CustomSequential(*layers))
                input_ch = ch * mult
            if level != 0:
                self.output_blocks.append(Upsample3D(input_ch))

        self.out = nn.Sequential(
            nn.GroupNorm(32, input_ch),
            nn.SiLU(),
            nn.Conv3d(input_ch, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, t, class_label=None):
        hs = []
        t_emb = get_timestep_embedding(t, self.ch)
        t_emb = self.time_embed(t_emb)

        if self.num_classes is not None and class_label is not None:
            c_emb = self.label_emb(class_label)
            t_emb = t_emb + c_emb

        h = x
        for module in self.input_blocks:
            if isinstance(module, CustomSequential):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)

        h = self.middle_block(h, t_emb)

        for module in self.output_blocks:
            if isinstance(module, Upsample3D):
                h = module(h)
            elif isinstance(module, CustomSequential):
                h_skip = hs.pop()
                h = torch.cat([h, h_skip], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)

        h = self.out(h)
        return h

import torch.utils.checkpoint as checkpoint

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout_layer = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        def _forward(x, t_emb):
            h = x
            h = self.norm1(h)
            h = F.silu(h)
            h = self.conv1(h)

            t = self.time_mlp(t_emb)
            h = h + t[:, :, None, None, None]

            h = self.norm2(h)
            h = F.silu(h)
            h = self.dropout_layer(h)
            h = self.conv2(h)

            return h + self.shortcut(x)

        if self.training:
            return checkpoint.checkpoint(_forward, x, t_emb)
        else:
            return _forward(x, t_emb)

class AttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q = q.reshape(B, C, D * H * W).permute(0, 2, 1)  # [B, N, C]
        k = k.reshape(B, C, D * H * W)  # [B, C, N]
        v = v.reshape(B, C, D * H * W).permute(0, 2, 1)  # [B, N, C]

        attn = torch.bmm(q, k) * (C ** (-0.5))
        attn = torch.softmax(attn, dim=2)
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, D, H, W)

        h = self.proj_out(h)
        return x + h

# Define Downsampling Layer
class Downsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# Define Upsampling Layer
class Upsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)

# Define DiffusionModule3D class
class DiffusionModule3D(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        B = x0.shape[0]

        t = torch.randint(0, self.var_scheduler.num_train_timesteps, (B,), device=self.device).long()

        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.var_scheduler.add_noise(x0, t, noise)

        predicted_noise = self.network(x_t, t, class_label)

        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 1, self.image_resolution, self.image_resolution, self.image_resolution]).to(self.device)

        traj = [x_T]
        for t in tqdm(reversed(range(self.var_scheduler.num_train_timesteps))):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x_t = traj[-1]

            noise_pred = self.network(x_t, t_batch, class_label)

            x_t_prev = self.var_scheduler.step(x_t, t_batch, noise_pred)

            traj.append(x_t_prev)

        if return_traj:
            return traj
        else:
            return traj[-1]

# Define Scheduler Class
class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=alphas_cumprod.device), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

        # sigmas correspond to σ_t in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when σ_t^2 = \tilde{β}_t.
            posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            sigmas = posterior_variance.sqrt()
        elif sigma_type == "large":
            # when σ_t^2 = β_t.
            sigmas = self.betas.sqrt()
        else:
            raise NotImplementedError(f"sigma_type {sigma_type} is not implemented.")

        self.register_buffer("sigmas", sigmas)

    def step(self, x_t: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.
        """
        betas_t = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1.0 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1, 1)
        sqrt_recip_alphas_t = (1.0 / self.alphas[t]).sqrt().view(-1, 1, 1, 1, 1)

        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        sigma_t = self.sigmas[t].view(-1, 1, 1, 1, 1)
        x_t_prev = model_mean + sigma_t * noise

        return x_t_prev

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).
        """
        if eps is None:
            eps = torch.randn_like(x_0)

        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_cumprod_t = alphas_cumprod_t.sqrt()
        sqrt_one_minus_alpha_cumprod_t = (1.0 - alphas_cumprod_t).sqrt()

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * eps

        return x_t

def get_data_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

def main(args):

    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    now = get_current_time()
    if args.use_cfg:
        save_dir = Path(f"results/cfg_diffusion-voxel-{now}")
    else:
        save_dir = Path(f"results/diffusion-voxel-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config.toDict(), f, indent=2)
    """######"""

    image_resolution = config.image_resolution
    category = args.category
    voxel_file = f"./data/hdf5_data/{category}_voxels_train_avgpool.npy"


    ds_module = VoxelDataModule(
        voxel_file=voxel_file,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        subset_size=config.max_num_images_per_cat,
        transform=None
    )
    ds_module.setup()

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )

    network = UNet3D(
        T=config.num_diffusion_train_timesteps,
        image_resolution=image_resolution,
        ch=64,
        ch_mult=[1, 2, 4, 8],
        attn=[],
        num_res_blocks=2,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=None,
    )

    ddpm = DiffusionModule3D(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0 and step > 0:
                ddpm.eval()

                fig_loss = plt.figure()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close(fig_loss)

                samples = ddpm.sample(4, return_traj=False)

                for i, sample in enumerate(samples):
                    sample_np = sample.cpu().numpy().squeeze(0)
                    img = visualize_voxel_prob(sample_np)
                    img.save(save_dir / f"step={step}-sample={i}.png")

                ddpm.train()

            voxel = next(train_it)  # [B, 1, D, H, W]
            voxel = voxel.to(config.device)
            optimizer.zero_grad()
            with autocast():
                if args.use_cfg:
                    loss = ddpm.get_loss(voxel)
                else:
                    loss = ddpm.get_loss(voxel)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

            pbar.set_description(f"Loss: {loss.item():.4f}")

            step += 1
            pbar.update(1)

    torch.save({
        'step': step,
        'model_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config.toDict(),
        'losses': losses,
    }, save_dir / f'{category}.ckpt')
    print("Saved Final Model Checkpoint")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=300000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=None,  
        help="max number of images per category",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=32)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--category", type=str, required=True, help="Category name (e.g., 'chair').")
    args, unknown = parser.parse_known_args()
    main(args)