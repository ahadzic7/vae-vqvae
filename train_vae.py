import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchvision import datasets, transforms
from torchvision.utils import save_image
from classes.MyVAE import MyVAE, MyOldVAE
from classes.modules import VAE
from torch.distributions.normal import Normal
from datetime import datetime
from utilities import *

torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader = data_setup()
model = MyVAE(input_dim=1, dim=256, latent_dim=128).to(device)  # 9099600
#model = MyOldVAE(input_dim=1, dim=256, latent_dim=128).to(device) # 9099601
opt = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

def vae_loss(x, m, beta=1.0):
    x_tilde, log_var, kl_div = m(x)
    rec_loss = -Normal(x_tilde, log_var.mul(0.5).exp()).log_prob(x).sum(dim=[2,3]).mean()
    return rec_loss + beta * kl_div
    
def train(device):
    train_loss = []
    model.train()
    for (x, _) in train_loader:
        loss = vae_loss(x.to(device), model)
        train_loss.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    return torch.tensor(train_loss).mean(0).item()

def test(device):
    val_loss = []
    model.eval()
    with torch.no_grad():
        for (x, _) in test_loader:
            loss = vae_loss(x.to(device), model)
            val_loss.append(loss.item())

    return torch.tensor(val_loss).mean(0).item()

N_EPOCHS = 51
BEST_LOSS = 99999
LAST_SAVED = -1
for epoch in range(N_EPOCHS):
    print(f"\nEpoch {epoch}:")
    tloss = train(device)
    print(f'Training loss: {tloss:.2f}')
    cur_loss = test(device)
    print(f'Testing loss: {cur_loss:.2f}')
    
    if LAST_SAVED == -1 or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model!")
        torch.save(model.state_dict(), f'models/VAE_good.pt')
    if epoch % 5 == 0:
        images = generate_recons(model, test_loader, device, nsamples=32)
        save_image(images, f'recons/VAE/VAE_good_rec_{epoch}.png', nrow=8)
        images = generate_samples(model, device, nsamples=64, dims=(128,1,1))
        save_image(images, f'samples/VAE/VAE_good_samp_{epoch}.png', nrow=8)


print(f"Last saved model: {LAST_SAVED}")
