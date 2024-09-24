import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from classes.modules import ResBlock, weights_init
from classes.modules import VQEmbedding
from classes.MyVAE import MyVAE

class MyVQVAE(nn.Module):
    def __init__(self, input_dim, dim, latent_dim, K):
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
            nn.Conv2d(dim, latent_dim, 3, 1, 0),
        )
        
        self.codebook = VQEmbedding(K, latent_dim)

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
        )
        self.apply(weights_init)

    def encode(self, x):
        return self.encoder(x).chunk(2, dim=1)

    def decode(self, z):
        return self.decoder(z).chunk(2, dim=1)

    def forward(self, x):
        z_e_x, _ = self.encode(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        mean, log_var = self.decode(z_q_x_st)
        return mean, log_var, z_e_x, z_q_x

class MyOldVQVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, 2*input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        return self.codebook(z_e_x)

    def decode(self, latents):
        return self.decoder(latents).chunk(2, dim=1)

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        mean, log_var = self.decode(z_q_x_st)
        return mean, log_var, z_e_x, z_q_x
    
 
class MyVQVAE2(nn.Module):
    def __init__(self, input_dim, dim, latent_dim, K):
        super().__init__()
        self.VAE = MyVAE(input_dim=input_dim, dim=dim, latent_dim=latent_dim)

        self.codebook = VQEmbedding(K, latent_dim)

        self.apply(weights_init)

    def encode(self, x):
        return self.VAE.encode(x)

    def decode(self, z):
        return self.VAE.decode(z)

    def forward(self, x):
        z_e_x, _ = self.encode(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        mean, log_var = self.decode(z_q_x_st)
        return mean, log_var, z_e_x, z_q_x
