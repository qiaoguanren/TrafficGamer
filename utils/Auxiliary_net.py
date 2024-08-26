import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(nn.Linear(state_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128))
        
        self.mu     = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        
        self.latent_mapping = nn.Linear(latent_dim, 128)
        
        self.decoder = nn.Sequential(nn.Linear(128, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, state_dim))
        
        self.f = nn.Linear(state_dim, 1)
        
        
    def encode(self, x):
        encoder = self.encoder(x)
        mu, logvar = self.mu(encoder), self.logvar(encoder)
        return mu, logvar
        
    def sample_z(self, mu, logvar):
        eps = torch.rand_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
    
    def decode(self, z):
        latent_z = self.latent_mapping(z)
        out = self.decoder(latent_z)
        reshaped_out = torch.sigmoid(out)
        return reshaped_out
        
    def forward(self, x):
        
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        output = self.decode(z)
        density = self.f(output)
        
        return density, output, mu, logvar
    
    def loss_function(self, output, state, mu, logvar, kld_weight):

        recon_loss = F.mse_loss(output, state)
        KLD = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim = 1), dim=0)
        return recon_loss + kld_weight * KLD
