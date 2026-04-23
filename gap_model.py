import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # A simple convolutional encoder for demonstration
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        # Assuming input image of 64x64
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.conv(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim=1):
        super(Decoder, self).__init__()
        # The decoder takes the latent vector PLUS the age condition
        self.fc = nn.Linear(latent_dim + condition_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),   # 32x32 -> 64x64
            nn.Sigmoid() # Output pixels between 0 and 1
        )

    def forward(self, z, age_condition):
        # Concatenate latent identity with age condition
        z_cond = torch.cat((z, age_condition), dim=1)
        hidden = self.fc(z_cond)
        hidden = hidden.view(-1, 128, 8, 8)
        out_img = self.deconv(hidden)
        return out_img

class GAP_CVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(GAP_CVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, age):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, age)
        return reconstructed, mu, logvar

    def simulate_age(self, x, target_age):
        """Used for inference when we want to change the age of an image"""
        device = next(self.parameters()).device
        x = x.to(device)
        target_age = target_age.to(device)
        
        # 1. Extract Identity Latent (mu)
        mu, _ = self.encoder(x)
        
        # 2. Decode with new target age
        projected_image = self.decoder(mu, target_age)
        return projected_image
