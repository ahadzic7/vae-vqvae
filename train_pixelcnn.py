import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from classes.MyVQVAE import MyVQVAE2, MyOldVQVAE
from classes.modules import GatedPixelCNN
from utilities import *
from experiment import pixelcnn_params

def prior_loss(vqvae, pixelcnn, batch_x, batch_y, k):
    with torch.no_grad():
        latents = vqvae.encode(batch_x).contiguous()
    logits = pixelcnn(latents, batch_y).permute(0, 2, 3, 1).contiguous()
    return F.cross_entropy(logits.view(-1, k), latents.view(-1))

def train(data_loader, vqvae, pixelcnn, opt, k, device):
    pixelcnn.train()
    l = []
    for batch_x, batch_y in data_loader:
        opt.zero_grad()
        loss = prior_loss(vqvae, pixelcnn, batch_x.to(device), batch_y.to(device), k)
        loss.backward()
        opt.step()
        l.append(loss.item())
    return torch.tensor(l).mean()

def test(data_loader, vqvae, pixelcnn, k, device):
    pixelcnn.eval()
    l = []
    for batch_x, batch_y in data_loader:
        loss = prior_loss(vqvae, pixelcnn, batch_x.to(device), batch_y.to(device), k)
        l.append(loss.item())
    return torch.tensor(l).mean()
    
def generate_samples(vqvae, pixelcnn, device, lat_dim, nsamples=64):
    return pixelcnn_params(vqvae, pixelcnn, device, lat_dim, nsamples)[0]
    	
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = data_setup()
    K=512
    
    type = "bad" # "bad"
    gm = MyOldVQVAE(1, dim=128, K=K).to(device)
    bm = MyVQVAE2(1, dim=256, latent_dim=128, K=K).to(device)
    vqvae, lat_dim = (gm,(7,7)) if type == "good" else (bm, (1,1))
    vqvae = load_model(vqvae, f"./saved_models/VQVAE_{type}.pt", device)
    vqvae.eval()

    pixelcnn = GatedPixelCNN(input_dim=512, dim=128, n_layers=15, n_classes=10).to(device)
    opt = torch.optim.Adam(pixelcnn.parameters(), lr=1e-3)

    best_loss = -1.
    for epoch in range(21):
        print(f"\nEpoch: {epoch}")
        tloss = train(train_loader, vqvae, pixelcnn, opt, K, device)
        print(f'Training loss: {tloss:.2f}')
        # The validation loss is not properly computed since the classes in the train and valid splits of Mini-Imagenet do not overlap.
        loss = test(test_loader, vqvae, pixelcnn, K, device)
        print(f'Testing loss: {loss:.2f}')

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(f'./models/prior_PCNN_{type}.pt', 'wb') as f:
                torch.save(pixelcnn.state_dict(), f)
                images = generate_samples(vqvae, pixelcnn, device, lat_dim, nsamples=64)
                save_image(images, f'samples/PCNN_VQVAE/VQVAE_{type}_samp_saved.png', nrow=8)
                print("Saving model!")
        if epoch % 5 == 0:
            images = generate_samples(vqvae, pixelcnn, device, lat_dim, nsamples=64)
            save_image(images, f'samples/PCNN_VQVAE/VQVAE_{type}_samp_{epoch}.png', nrow=8)
        
if __name__ == '__main__':
    main()
