import torch
from torch.distributions import Normal
from classes.modules import VectorQuantizedVAE, GatedPixelCNN
from classes.MyVAE import MyVAE
from utilities import *
# srun -p gpufast --gres=gpu:1 --pty bash -i

def pixelcnn_dist(vqvae, pixelcnn, device, nsamples=2**7):
    classes = torch.randint(low=0, high=10, size=(nsamples,)).to(device)
    z = pixelcnn.generate(classes, batch_size=nsamples)
    z_embedded = vqvae.codebook.embedding(z).permute(0, 3, 1, 2) # (B, D, H, W)
    return vqvae.decode(z_embedded)
    
def vae_params(vae, device, i=14):
    samples = torch.randn(2**i, 128, 1, 1).to(device)
    mean, log_var = vae.decode(samples)
    return mean, log_var.mul(0.5).exp()

def vae_params2(vae, samps, device):
    mean, log_var, _ = vae(samps)
    return mean, log_var.mul(0.5).exp()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = data_setup()
    vae = MyVAE(input_dim=1, dim=256, latent_dim=128)
    vae = load_model(vae, "./saved_models/VAE.pt", device)

    for i in range(1, 15):
        nsamples=2**i
        mixture = Normal(*vae_params(vae, device, i=i))
        bpd = torch.cat([bpd_batch(mixture, batch_x.to(device), i=i) for batch_x,_ in train_loader])
        print(f"{i} Min:{bpd.min():.2f} - Mean:{bpd.mean():.2f} - Max:{bpd.max():.2f}\n")

    # nsamples=2**14
    # mixture = Normal(*vae_params(vae, device, i=14))
    # bpd = torch.cat([bpd_batch(mixture, batch_x.to(device), i=14) for batch_x,_ in test_loader])

    print(f" Min:{bpd.min():.2f} - Mean:{bpd.mean():.2f} - Max:{bpd.max():.2f}\n")

if __name__ == '__main__':
    torch.manual_seed(1)
    main()


