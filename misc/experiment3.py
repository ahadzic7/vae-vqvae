from models2.cm import ContinuousMixture, GaussianDecoder
from torchvision.datasets import MNIST, FashionMNIST
from utils2.bins_samplers import GaussianQMCSampler
from utils2.reproducibility import seed_everything
from utils2.datasets import UnsupervisedDataset
import torchvision.transforms as transforms
from models2.nets import mnist_conv_decoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchvision import transforms, datasets


def load_model(path, device):
    model = ContinuousMixture.load_from_checkpoint(path).to(device)
    model.missing = False
    model.eval()
    return model


def data_setup():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./mnist", train=False, transform=transform)
    return train_dataset.data.cuda(), test_dataset.data.cuda()


latent_dim = 16
n_filters = 16
batch_norm = True
mu_activation = nn.Sigmoid()
bias = False
resblock = True
learn_std = True
min_std = 0.1
max_std = 1.0
n_bins = 2**14

seed_everything(0)
mcd = mnist_conv_decoder(latent_dim=latent_dim,
                         n_filters=n_filters,
                         batch_norm=batch_norm,
                         learn_std=learn_std,
                         bias=bias,
                         resblock=resblock)
gaussdecoder = GaussianDecoder(mcd, learn_std, min_std, max_std, mu_activation)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(path="/home/hadziarm/logs/mnist/cm/version_28/checkpoints/best_model_valid-epoch=27.ckpt", device=device)

model.n_chunks = 32
model.sampler.n_bins = 2**14
#z, log_w = model.sampler(seed=42)

def bpd(cm_model, data):
    lls = cm_model.eval_loader(data, progress_bar=True)
    bpd = -lls.sum() / (10000 * 784 * torch.log(torch.tensor(2.0)))
    return bpd.item()

train_data, test_data = data_setup()
print(bpd(model, test_data))

# dataset, dataset_name = FashionMNIST, 'fashion_mnist'
# dataset, dataset_name = MNIST, 'mnist'
# transf = transforms.Compose([transforms.ToTensor()])
# test = UnsupervisedDataset(dataset(root='./data/', train=False, download=True, transform=transf))
# test_loader = DataLoader(test, batch_size=128)
# print('Computing test LL using %d bins..' % n_bins)
# print(model.eval_loader(gaussian_samples, z, log_w, device=device).mean().item())
# latent_dim = model.sampler.latent_dim
# samples = model.decoder.net(torch.randn(16, latent_dim).to(device)).detach().cpu()
# print(samples.shape)
# grid_img = torchvision.utils.make_grid(samples.view(16, 2, 28, 28), nrow=4)

# from PIL import Image

# image_np = grid_img.permute(1, 2, 0).numpy()  # Change channel ordering for PIL (HWC format)

# # Convert the NumPy array to a PIL Image
# image_pil = Image.fromarray((image_np * 255).astype('uint8'))  # Scale to 0-255

# # Save the image
# image_pil.save('image_grid2.png')
