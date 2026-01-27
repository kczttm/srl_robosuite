import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import math

class ResNetBlock(nn.Module):
    """ResNet block with configurable GroupNorm groups"""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        # Ensure groups divide output channels
        assert out_channels % groups == 0, f"out_channels {out_channels} must be divisible by groups {groups}"
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        return x + residual
    
class Encoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=2):
        super().__init__()
        # Input: (batch, 2, 16, 16)
        self.conv_in = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.res1 = ResNetBlock(4, 8, groups=8)     # 8/8=1 group
        self.down1 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 16x16 → 8x8
        self.res2 = ResNetBlock(16, 16, groups=8)   # 16/8=2 groups
        self.down2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 8x8 → 4x4
        self.res3 = ResNetBlock(32, 32, groups=8)   # 32/8=4 groups
        self.mean = nn.Conv2d(32, latent_dim, kernel_size=1)
        self.logvar = nn.Conv2d(32, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)       # 4 channels
        x = self.res1(x)          # 8 channels
        x = self.down1(x)          # 16 channels
        x = self.res2(x)          
        x = self.down2(x)          # 32 channels
        x = self.res3(x)          
        mean, logvar = self.mean(x), self.logvar(x)
        z = mean + torch.exp(0.5*logvar) * torch.randn_like(logvar)
        return z, mean, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels=2, latent_dim=2):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, 32, kernel_size=3, padding=1)
        self.res1 = ResNetBlock(32, 32, groups=8)    # 32/8=4 groups
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, 
                                     padding=1, output_padding=1)  # 4x4 → 8x8
        self.res2 = ResNetBlock(16, 16, groups=8)     # 16/8=2 groups
        self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)  # 8x8 → 16x16
        self.res3 = ResNetBlock(8, 4, groups=4)       # 4/4=1 group (critical fix)
        self.conv_out = nn.Conv2d(4, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)       # 32 channels
        x = self.res1(x)          
        x = self.up1(x)           # 16 channels
        x = self.res2(x)          
        x = self.up2(x)           # 8 channels
        x = self.res3(x)          # 4 channels
        return self.conv_out(x)   # 2 channels

class AutoencoderKL(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(out_channels, latent_dim)

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_x, kl_loss
    
    def recons_from_feature(self, z):
        recon_x = self.decoder(z)
        return recon_x

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=2, input_size=16, target_size=2):
        super().__init__()

        layers = []
        channels = in_channels
        hidden = 8

        # Compute number of downsamples
        num_downsamples = int(math.log2(input_size // target_size))

        for _ in range(num_downsamples):
            layers.append(nn.Conv2d(channels, hidden, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            channels = hidden
            hidden *= 2

        # Project to latent_dim channels
        layers.append(nn.Conv2d(channels, latent_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)  # shape: [B, 2, target_size, target_size]
    
class AutoDecoder(nn.Module):
    def __init__(self, out_channels=2, latent_dim=2, target_size=2, output_size=16, config=None):
        super().__init__()

        layers = []
        channels = latent_dim
        hidden = 64

        num_upsamples = int(math.log2(output_size // target_size))

        for _ in range(num_upsamples):
            layers.append(nn.ConvTranspose2d(channels, hidden, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
            channels = hidden
            hidden //= 2

        layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

        if config is not None:
            self.image_width = config['image_shape']['image_width']
            self.image_height = config['image_shape']['image_height']
        else:
            self.image_width = 640
            self.image_height = 360

    def forward(self, z):
        x = self.decoder(z)
        x = torch.relu(x)
        x = torch.stack([
            torch.clamp(x[:, 0], 0, self.image_width),
            torch.clamp(x[:, 1], 0, self.image_height)
        ], dim=1)
        return x # shape: [B, 2, output_size, output_size]
    
class Autoencoder_V2(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, latent_dim=2, config=None):
        super().__init__()
        self.sqrt_flow_points = int(np.sqrt(config['model']['num_flows']))
        
        self.encoder = AutoEncoder(in_channels, latent_dim, input_size=self.sqrt_flow_points, target_size=config['model']['latent_dim'])
        self.decoder = AutoDecoder(out_channels, latent_dim, target_size=config['model']['latent_dim'], output_size=self.sqrt_flow_points, config=config)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
    
    def recons_from_feature(self, z):
        recon_x = self.decoder(z)
        return recon_x
    
class InMemoryNumpyArrayDataset(Dataset):
    def __init__(self, file_paths, config):
        self.data = []  # List of tensors: each (T, num_flow, 2) 
        self.vae_v2 = config['model']['vae_v2']
        for file_path in file_paths:
            arrays = np.load(file_path)  # shape: (1, T, num_flow, 2)
            frames = arrays[0]  # shape: (T, num_flow, 2) | each element is (u,v) image coordinate
            frames = frames.astype(np.float32)
            if not self.vae_v2:  # if using vae_V2, no need for normalization
                frames[..., 0] /= config['image_shape']['image_width']  # Normalize x by image width
                frames[..., 1] /= config['image_shape']['image_height']  # Normalize y by image height
            self.data.append(frames)
        
        self.data = np.concatenate(self.data, axis=0)  # shape: (total_T, num_flow_point, 2)
        self.number_flow_point = self.data.shape[1]
        self.reshape_number = int(np.sqrt(self.number_flow_point))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        frame = self.data[idx].reshape(self.reshape_number, self.reshape_number, 2).transpose(2, 0, 1)  # (2, self.reshape_number, self.reshape_number)
        tensor = torch.from_numpy(frame)
        return tensor, tensor  # For autoencoder

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(subfolders, config, save_dir):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    set_seed(config['training']['seed'])

    if config['model']['vae_v2']:
        model = Autoencoder_V2(config=config).to(device) 
    else:
        model = AutoencoderKL().to(device)  
    
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, "autoencoder_models", date_string)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path + "/init_model.pth")

    all_files = []

    for subfold in subfolders:
        path = subfold + '/obj_flow_traj.npy'
        if os.path.exists(path):
            all_files.append(path)

    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)  # 70% trajs are used for training
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    batch_size = config['training']['batch_size']
    train_dataset = InMemoryNumpyArrayDataset(train_files, config)
    val_dataset = InMemoryNumpyArrayDataset(val_files, config)
    test_dataset = InMemoryNumpyArrayDataset(test_files, config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = optim.Adam(
        model.parameters(),
        lr = float(config['training']['learning_rate']),
        betas = config['training']['beta'],
        weight_decay = float(config['training']['weight_decay']),
        eps = float(config['training']['eps'])
    )

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(config['training']['gamma']))

    train_losses = []
    val_losses = []

    num_epoch = int(config['training']['num_epochs'])
    for epoch in tqdm(range(num_epoch), desc="Training Epochs"):
        # Training
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:  # inputs and targets are the same
            inputs = inputs.to(device)
            optimizer.zero_grad()
            if config['model']['vae_v2']:  # autoencoder V2 does not require the KL loss
                reconstructions = model(inputs)
                loss = F.mse_loss(reconstructions, inputs)
            else:
                reconstructions, kl_loss = model(inputs)
                loss = F.mse_loss(reconstructions, inputs) + 0.5 * kl_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                if config['model']['vae_v2']:
                    reconstructions = model(inputs)
                    val_loss += (F.mse_loss(reconstructions, inputs) ).item()
                else:
                    reconstructions, kl_loss = model(inputs)
                    val_loss += (F.mse_loss(reconstructions, inputs) + 0.5 * kl_loss).item()
                # val_loss += (F.mse_loss(reconstructions, inputs)).item()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}: Train Loss {avg_train:.4f} | Val Loss {avg_val:.4f}")
        
        # # Early Stopping
        # if avg_val < best_val_loss:
        #     best_val_loss = avg_val
        #     no_improve = 0
        #     torch.save(model.state_dict(), "best_model.pth")
        # else:
        #     no_improve += 1
        #     if no_improve == patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break

    torch.save(model.state_dict(), save_path + "/final_model.pth")

    # Save a copy
    with open(save_path + "/autoencoder_config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epoch + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epoch + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Optional: ensures labels don't get cut off
    plt.savefig(save_path + "/loss_plot.png")  # <- Correct saving
    plt.show()

import argparse

if __name__ == '__main__':
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Find all subfolders under a folder.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory")
    parser.add_argument("--config_path", type=str, required=True, help="path to the training config file")
    parser.add_argument("--save_dir", type=str, required=True, help="path to save the trained models")

    args = parser.parse_args()
    root = Path(args.root_dir)

    # Get immediate subfolders
    subfolders = [str(f) for f in root.iterdir() if f.is_dir() and 'flow_data' in f.name]
    subfolders = sorted(subfolders)
    
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Access config values
    train(subfolders, config, args.save_dir)




        

