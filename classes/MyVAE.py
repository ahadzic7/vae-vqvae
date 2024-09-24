import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from classes.modules import ResBlock, weights_init
from classes.modules import VQEmbedding

class MyVAE(nn.Module):
    def __init__(self, input_dim, dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 2*latent_dim, 3, 1, 0),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, 2*input_dim, 4, 2, 1),
            nn.Tanh()
            # nn.Sigmoid()
        )

        self.apply(weights_init)

    def encode(self, x):
        #z_e_x = self.encoder(x)
        #return self.fc_mu(z_e_x), self.fc_logvar(z_e_x) 
        return self.encoder(x).chunk(2, dim=1)

    def decode(self, z):
        out = self.decoder(z)
        return out.chunk(2, dim=1)

    def forward(self, x):
        mu, log_var = self.encode(x)
        
        q_z_x = Normal(mu, log_var.mul(0.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        
        x_tilde, log_var = self.decode(q_z_x.rsample())
        return x_tilde, log_var, kl_div


class MyOldVAE(nn.Module):
    def __init__(self, input_dim, dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )
        
        self.fc_mu = nn.Conv2d(dim, latent_dim,  1, 1, 0)
        self.fc_logvar = nn.Conv2d(dim, latent_dim, 1, 1, 0)

        self.decoder = nn.Sequential(
            ResBlock(latent_dim),
            ResBlock(latent_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(latent_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, 2 * input_dim, 4, 2, 1),
            nn.Tanh(), #nn.Sigmoid()
        )
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        return self.fc_mu(z_e_x), self.fc_logvar(z_e_x) 

    def decode(self, z):
        out = self.decoder(z)
        return out.chunk(2, dim=1)

    def forward(self, x):
        mu, log_var = self.encode(x)
        
        q_z_x = Normal(mu, log_var.mul(0.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        
        x_tilde, log_var = self.decode(q_z_x.rsample())
        return x_tilde, log_var, kl_div
