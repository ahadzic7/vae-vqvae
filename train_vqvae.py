# python vqvae.py --data-folder ./mnist --output-folder vqvae
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from classes.MyVQVAE import MyVQVAE, MyOldVQVAE, MyVQVAE2
from utilities import *

def vqvae_loss(x, m, beta=1.0):
    x_tilde, log_var, z_e_x, z_q_x = m(x)
    loss_recons = -Normal(x_tilde, log_var.mul(0.5).exp()).log_prob(x).sum(dim=[2,3]).mean()
    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
    return loss_recons + loss_vq + beta * loss_commit

def train(data_loader, model, opt, device):
    train_loss = []
    for x, _ in data_loader:
        loss = vqvae_loss(x.to(device), model) 
        train_loss.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.tensor(train_loss).mean().item()        

def test(data_loader, model, device):
    model.eval()
    val_loss = []
    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            loss = vqvae_loss(x.to(device), model)
            val_loss.append(loss.item())
    return torch.tensor(val_loss).mean(0)

def generate_recons(model, dataloader, device, nsamples):
    model.eval()
    indices = torch.randperm(len(dataloader.dataset))[:nsamples]
    x = torch.stack([dataloader.dataset[i][0] for i in indices]).to(device) 
    x_tilde, _, _, _ = model(x)
    x_cat = torch.cat([x, x_tilde], 0).cpu()
    images = (x_cat + 1) / 2
    return images.to(device)

def main():
    # empty_folder(folder_path=r"./recons/VQVAE/")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = data_setup()
    # model = MyVQVAE(input_dim=1, dim=256, latent_dim=128, K=512).to(device)
    # model = MyOldVQVAE(1, dim=128, K=512).to(device)
    model = MyVQVAE2(input_dim=1, dim=256, latent_dim=128, K=512).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = -1.
    for epoch in range(51):
        print(f'\nEpoch {epoch}')
        tloss = train(train_loader, model, opt, device)
        print(f'Training loss: {tloss:.2f}')
        loss = test(test_loader, model, device)
        print(f'Test loss:{loss:.2f}')

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            print("Saving model!")
            with open(f'./models/VQVAE_bad.pt', 'wb') as f:
                torch.save(model.state_dict(), f)
            images = generate_recons(model, test_loader, device, nsamples=32)
            save_image(images, f'recons/VQVAE/VQVAE_bad_rec_saved.png', nrow=8)
        if epoch % 5 == 0:
            images = generate_recons(model, test_loader, device, nsamples=32)
            save_image(images, f'recons/VQVAE/VQVAE_bad_rec_{epoch}.png', nrow=8)


if __name__ == '__main__':
    main()
