import torch
from experiment import pixelcnn_params
from classes.modules import VectorQuantizedVAE, GatedPixelCNN
from classes.MyVAE import MyVAE, MyOldVAE
from classes.MyVQVAE import MyOldVQVAE, MyVQVAE2
from utilities import *

def prepare_mixtures(vqvae, pxcnn, device, lat_dim, ex, file_name):
    N = 11
    for idx in range(2**(ex - N)):
        print(idx)
        loc, scale = pixelcnn_params(vqvae, pxcnn, device, lat_dim, i=N)
        torch.save({'loc': loc, 'scale': scale}, f'{file_name}_{idx}.pth')

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = data_setup()

    vqvae_good = MyOldVQVAE(1, dim=128, K=512).to(device)
    vqvae_good = load_model(vqvae_good, f"./saved_models/VQVAE_good.pt", device)
    vqvae_bad = MyVQVAE2(1, dim=256, latent_dim=128, K=512).to(device)
    vqvae_bad = load_model(vqvae_bad, f"./saved_models/VQVAE_bad.pt", device)

    pixelcnn = GatedPixelCNN(input_dim=512, dim=128, n_layers=15, n_classes=10)
    pixelcnn_good = load_model(pixelcnn, f"./models/prior_PCNN_good.pt", device)
    pixelcnn_bad = load_model(pixelcnn, f"./models/prior_PCNN_bad.pt", device)

    prepare_mixtures(vqvae_good, pixelcnn_good, device, lat_dim=(7,7), ex=14, file_name='./mixtures/good/Mix_good') # 2^ex centers
    prepare_mixtures(vqvae_bad, pixelcnn_bad, device, lat_dim=(1,1), ex=14,  file_name= './mixtures/bad/Mix_bad')