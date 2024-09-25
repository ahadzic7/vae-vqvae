import torch
from torch.distributions import Normal
from classes.modules import VectorQuantizedVAE, GatedPixelCNN
from classes.MyVAE import MyVAE, MyOldVAE
from classes.MyVQVAE import MyOldVQVAE, MyVQVAE2
from utilities import *
import tqdm
# srun -p gpufast --gres=gpu:1 --pty bash -i

def pixelcnn_params(vqvae, pixelcnn, device, lat_dim, i):
    classes = torch.randint(low=0, high=10, size=(2**i,)).to(device)
    z = pixelcnn.generate(classes, batch_size=2**i, shape=lat_dim)
    z_embedded = vqvae.codebook.embedding(z).permute(0, 3, 1, 2) # (B, D, H, W)
    mean, log_var = vqvae.decode(z_embedded)
    return mean, log_var.mul(0.5).exp()
    
def vae_params(vae, device, lat_dim, i):
    samples = torch.randn(2**i, *lat_dim).to(device)
    mean, log_var = vae.decode(samples)
    return mean, log_var.mul(0.5).exp()

def vae_params2(vae, samps):
    mean, log_var, _ = vae(samps)
    return mean, log_var.mul(0.5).exp()

def vae_result(vae_good, vae_bad, train_loader, test_loader, device, i=14):
    print("Good sampling")
    mixture = Normal(*vae_params(vae_good, device, lat_dim=(128, 1, 1), i=i)) # 2^i
    bpd_train = torch.cat([bpd_batch(mixture, batch_x.to(device), i) for batch_x,_ in train_loader])
    bpd_test = torch.cat([bpd_batch(mixture, batch_x.to(device), i) for batch_x,_ in test_loader])
    print(f"Train: {bpd_train.mean():.2f} Test: {bpd_test.mean():.2f}\n")

    print("Bad sampling")
    mixture = Normal(*vae_params(vae_bad, device, lat_dim=(128, 7, 7), i=14)) # 2^i
    bpd_train = torch.cat([bpd_batch(mixture, batch_x.to(device), i=14) for batch_x,_ in train_loader])
    bpd_test = torch.cat([bpd_batch(mixture, batch_x.to(device), i=14) for batch_x,_ in test_loader])
    print(f"Train: {bpd_train.mean():.2f} Test: {bpd_test.mean():.2f}\n")

def batch2(x, device, type):
    log2 = torch.tensor(2).log().to(device)
    ttt = torch.empty(2**14, device=device)

    for idx in range(8):
        params = torch.load(f'./mixtures/{type}/Mix_{type}_{idx}.pth', weights_only=True)
        mixture = Normal(params['loc'], params['scale'])
        ll_px = mixture.log_prob(x) - log2.mul(14)
        ll_im = ll_px.sum(dim=[2, 3])
        s_idx = idx * 2**11
        e_idx = s_idx + 2**11
        
        ttt[s_idx:e_idx] = ll_im.squeeze(1)
    return -ttt.logsumexp(dim=0).div(784).div(log2) 

def help(batch, device, type):
    return torch.tensor([batch2(bx.to(device), device, type) for bx in batch[0]])

def vqvae_result2(train_loader, test_loader, device):
    print("Good sampling")
    bpd_train = torch.cat([help(bx, device, type="good") for bx,_ in train_loader])
    bpd_test = torch.cat([help(bx, device, type="good") for bx,_ in test_loader])
    print(f"Train: {bpd_train.mean():.2f} Test: {bpd_test.mean():.2f}\n")
    
    print("Bad sampling")
    bpd_train = torch.cat([help(bx, device, type="bad") for bx,_ in train_loader])
    bpd_test = torch.cat([help(bx, device, type="bad") for bx,_ in test_loader])
    print(f"Train: {bpd_train.mean():.2f} Test: {bpd_test.mean():.2f}\n")

    
def vqvae_result(vqvae_good, pxcnn_good, vqvae_bad, pxcnn_bad, train_loader, test_loader, device, i):
    print("Good sampling")
    mixture = Normal(*pixelcnn_params(vqvae_good, pxcnn_good, device, lat_dim=(7, 7), i=i)) # 2^i
    bpd_train = torch.cat([bpd_batch(mixture, batch_x.to(device), i=i) for batch_x,_ in train_loader])
    print(".")
    bpd_test = torch.cat([bpd_batch(mixture, batch_x.to(device), i=i) for batch_x,_ in test_loader])
    print(f"Train: {bpd_train.mean():.2f} Test: {bpd_test.mean():.2f}\n")

    print("Bad sampling")
    mixture = Normal(*pixelcnn_params(vqvae_bad, pxcnn_bad, device, lat_dim=(1, 1), i=i)) # 2^i
    bpd_train = torch.cat([bpd_batch(mixture, batch_x.to(device), i=i) for batch_x,_ in train_loader])
    print(".")
    bpd_test = torch.cat([bpd_batch(mixture, batch_x.to(device), i=i) for batch_x,_ in test_loader])
    print(f"Train: {bpd_train.mean():.2f} Test: {bpd_test.mean():.2f}\n")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = data_setup()
    vae_good = MyVAE(input_dim=1, dim=256, latent_dim=128)
    vae_good = load_model(vae_good, "./saved_models/VAE_good.pt", device)
    vae_bad = MyOldVAE(input_dim=1, dim=256, latent_dim=128)
    vae_bad = load_model(vae_bad, "./saved_models/VAE_bad.pt", device)

    vqvae_good = MyOldVQVAE(1, dim=128, K=512)
    vqvae_good = load_model(vqvae_good, f"./saved_models/VQVAE_good.pt", device)
    vqvae_bad = MyVQVAE2(1, dim=256, latent_dim=128, K=512)
    vqvae_bad = load_model(vqvae_bad, f"./saved_models/VQVAE_bad.pt", device)

    pxcnn = GatedPixelCNN(input_dim=512, dim=128, n_layers=15, n_classes=10)
    pxcnn_good = load_model(pxcnn, f"./models/prior_PCNN_good.pt", device)
    pxcnn_bad = load_model(pxcnn, f"./models/prior_PCNN_bad.pt", device)

    

    print("VAE")
    vae_result(vae_good, vae_bad, train_loader, test_loader, device, i=14)
    print("VQVAE")
    vqvae_result2(train_loader, test_loader, device)
    # vqvae_result(vqvae_good, pxcnn_good, vqvae_bad, pxcnn_bad, train_loader, test_loader, device, i=11)

    
    

    # for i in range(1, 15):
    #     nsamples=2**i
    #     mixture = Normal(*vae_params(vae, device, i=i))
    #     bpd = torch.cat([bpd_batch(mixture, batch_x.to(device), i=i) for batch_x,_ in train_loader])
    #     print(f"{i} Min:{bpd.min():.2f} - Mean:{bpd.mean():.2f} - Max:{bpd.max():.2f}\n")

if __name__ == '__main__':
    torch.manual_seed(1)
    main()


