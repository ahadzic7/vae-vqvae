import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.distributions import Normal
from torch.utils.data import DataLoader
import os

def bits_per_dim(mixture, x, i):
    log2 = torch.tensor(2).log()
    ll_px = mixture.log_prob(x) - log2.mul(i)
    ll_im = ll_px.sum(dim=[2,3])
    mix_model = ll_im.logsumexp(dim=0)
    return -mix_model / log2.mul(784)

def bpd_batch(mixture, batch, i):
    return torch.tensor([bits_per_dim(mixture, x, i).item() for x in batch])


def generate_recons(model, dataloader, device, nsamples):
    model.eval()
    indices = torch.randperm(len(dataloader.dataset))[:nsamples]
    x = torch.stack([dataloader.dataset[i][0] for i in indices]).to(device) 
    x_tilde, _, _ = model(x)
    x_cat = torch.cat([x, x_tilde], 0).cpu()
    images = (x_cat + 1) / 2
    return images.to(device)


def generate_samples(model, device, nsamples, dims=(128, 1, 1)):
    model.eval()
    z_e_x = Normal(0, 1).sample((nsamples, *dims)).to(device)
    x_tilde, _ = model.decode(z_e_x)
    images = (x_tilde.cpu() + 1) / 2
    return images.to(device)


 
def empty_folder(folder_path):
    for filename in os.listdir(folder_path): 
        file_path = os.path.join(folder_path, filename)  
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  
            elif os.path.isdir(file_path):  
                os.rmdir(file_path)  
        except Exception as e:  
            print(f"Error deleting {file_path}: {e}")
    print("Deletion done")

def load_model(model, model_file, device):
    cp = torch.load(model_file, weights_only=True)
    model.load_state_dict(cp)
    return model.to(device)


def data_setup(batch_size=128):
    preproc = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds_train = MNIST("./mnist", train=True, download=False, transform=preproc,)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ds_test = MNIST("./mnist", train=False, transform=preproc)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

