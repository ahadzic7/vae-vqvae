import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from classes.MyVAE import MyVQVAE2
from classes.modules import VectorQuantizedVAE, GatedPixelCNN
from misc.datasets import MiniImagenet
from torchvision import transforms, datasets
from utilities import *

def train(data_loader, model, prior, optimizer, args, writer):
    for images, labels in data_loader:
        with torch.no_grad():
            images = images.to(args.device)
            latents = model.encode(images)
            latents = latents.detach()

        labels = labels.to(args.device)
        logits = prior(latents, labels)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, args.k), latents.view(-1))
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, prior, args, writer):
    with torch.no_grad():
        loss = 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            latents = model.encode(images)
            latents = latents.detach()
            logits = prior(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss += F.cross_entropy(logits.view(-1, args.k), latents.view(-1))

        loss /= len(data_loader)

    # Logs
    writer.add_scalar('loss/valid', loss.item(), args.steps)

    return loss.item()

def data_setup(args):
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        num_channels = 1
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(args.data_folder, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False, transform=transform)
        elif args.dataset == 'fashion-mnist':
            train_dataset = datasets.FashionMNIST(args.data_folder, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder, train=False, transform=transform)
        elif args.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(args.data_folder, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder, train=False, transform=transform)
            num_channels = 3
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True, download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True, download=True, transform=transform)
        num_channels = 3

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               num_workers=args.num_workers, 
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               drop_last=True, 
                                               num_workers=args.num_workers, 
                                               pin_memory=True)

    return train_loader, test_loader, num_channels

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, num_channels = data_setup(device)

    vqvae = MyVQVAE2(1, dim=256, latent_dim=128, K=512).to(device)
    vqvae = load_model(vqvae, "./saved_models/VQVAE.pt")
    vqvae.eval()

    prior = GatedPixelCNN(args.k, args.hidden_size_prior, args.num_layers, n_classes=10).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=1e-3)

    best_loss = -1.
    for epoch in range(args.num_epochs):
        print(epoch)
        train(train_loader, vqvae, prior, opt)
        # The validation loss is not properly computed since the classes in the train and valid splits of Mini-Imagenet do not overlap.
        loss = test(test_loader, vqvae, prior, args)
        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(f'./models/{args.output_folder}/prior.pt', 'wb') as f:
                torch.save(prior.state_dict(), f)

if __name__ == '__main__':
    main()
