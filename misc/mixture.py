import sys
import os
from models2.cm import ContinuousMixture, GaussianDecoder
from torchvision.datasets import MNIST, FashionMNIST
from utils2.bins_samplers import GaussianQMCSampler
from utils2.bins_samplers import GaussianSampler
from utils2.reproducibility import seed_everything
from utils2.datasets import UnsupervisedDataset
import torchvision.transforms as transforms
from models2.nets import mnist_conv_decoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch

repo_dir = os.path.dirname(os.getcwd())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = None if device == 'cpu' else 1
print(device)

dataset, dataset_name = MNIST, 'mnist'

transf = transforms.Compose([transforms.ToTensor()])

train = UnsupervisedDataset(dataset(root=repo_dir + '/data', train=True, download=True, transform=transf))
train, valid = torch.utils.data.random_split(train, [50_000, 10_000])

batch_size = 256
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid, batch_size=batch_size)

max_epochs = 100
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
# sampler = GaussianSampler(latent_dim,n_bins)
sampler = GaussianQMCSampler(latent_dim, n_bins)
model = ContinuousMixture(gaussdecoder, sampler=sampler)
model.n_chunks = 32
model.missing = None

cp_best_model_valid = pl.callbacks.ModelCheckpoint(save_top_k=1,
                                                   monitor='valid_loss_epoch',
                                                   mode='min',
                                                   filename='best_model_valid-{epoch}')
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="valid_loss_epoch",
                                                                min_delta=0.0,
                                                                patience=15,
                                                                verbose=False,
                                                                mode='min')
callbacks = [cp_best_model_valid, early_stop_callback]

print(repo_dir + '/logs/' + dataset_name)

logger = pl.loggers.TensorBoardLogger(repo_dir + '/logs/' + dataset_name, name='cm')
trainer = pl.Trainer(max_epochs=max_epochs, callbacks=callbacks, logger=logger, deterministic=True)

trainer.fit(model, train_loader, valid_loader)
