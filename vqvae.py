# python vqvae.py --data-folder ./mnist --output-folder vqvae
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal
from classes.modules import VectorQuantizedVAE, to_scalar
from misc.datasets import MiniImagenet
from tensorboardX import SummaryWriter

def vae_loss(x, model, beta=1.0):
    x_tilde, log_var, kl_div = model(x)
    nll = -Normal(x_tilde, torch.sqrt(torch.exp(log_var))).log_prob(x)
    reconstruction_loss = nll.sum() / x.size(0)
    loss = reconstruction_loss + beta * kl_div
    return loss

def train(data_loader, model, optimizer, args, writer):
    for x, _ in data_loader:
        x = x.to(args.device)

        optimizer.zero_grad()
        # x_tilde, z_e_x, z_q_x = model(images)
        x_tilde, log_var, z_e_x, z_q_x = model(x)
        
        # Convert log-variance to variance
        var = torch.exp(log_var)

        # Reconstruction loss (Negative Log-Likelihood)
        # NLL for each pixel: (log_var + (x - mean)^2 / variance)
        # loss_recons = 0.5 * (log_var + ((images - x_tilde) ** 2) / var)
        # loss_recons = loss_recons.mean()  # Take the mean across all pixels
        nll = -Normal(x_tilde, torch.sqrt(torch.exp(log_var))).log_prob(x)
        loss_recons = nll.mean()
        
        # Vector quantization objective (as before)
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective (as before)
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        # Combine the losses (adding VQ-VAE loss components)
        loss = loss_recons + loss_vq + args.beta * loss_commit

        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, args, writer):
    loss = 0.
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(args.device)
            # x_tilde, z_e_x, z_q_x = model(images)
            x_tilde, log_var, z_e_x, z_q_x = model(images)
            # Convert log-variance to variance
            var = torch.exp(log_var)  # variance = e^(log_var)

            # Reconstruction loss (Negative Log-Likelihood)
            # NLL for each pixel: (log_var + (x - mean)^2 / variance)
            nll = -Normal(x_tilde, torch.sqrt(torch.exp(log_var))).log_prob(x)
            loss_recons = nll.mean()
            
            # Vector quantization objective (as before)
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

            # Commitment objective (as before)
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

            # Combine the losses (adding VQ-VAE loss components)
            loss += loss_recons + loss_vq + args.beta * loss_commit

    return loss / len(data_loader)

def evaluate(model, data_loader, args):
    total_loss = 0.0
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculations
        for images, _ in data_loader:
            images = images.to(args.device)

            # Forward pass through the VQ-VAE model
            x_tilde, log_var, z_e_x, z_q_x = model(images)

            # Convert log-variance to variance
            # Clamp log-variance to prevent extremely small variance (avoid numerical instability)
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            var = torch.exp(log_var)  # variance = e^(log_var)

            # Reconstruction loss (Negative Log-Likelihood)
            # NLL for each pixel: 0.5 * (log_var + (x - mean)^2 / variance)
            loss_recons = 0.5 * (log_var + ((images - x_tilde) ** 2) / var)
            loss_recons = loss_recons.mean()  # Take the mean across all pixels

            # Vector quantization objective (as in training loop)
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

            # Commitment objective (as in training loop)
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

            # Combine the losses
            total_loss = (loss_recons + loss_vq + args.beta * loss_commit).item()
            # print(total_loss)
            # print()

    # Return the average loss over the dataset
    return total_loss / len(data_loader)

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _, _ = model(images)
    return x_tilde

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
    writer = SummaryWriter(f'./logs/{args.output_folder}')
    save_filename = f'./models/{args.output_folder}'

    train_loader, test_loader, num_channels = data_setup(args)
    
    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer)
        loss = test(test_loader, model, args, writer)
        
        print(f'Epoch:{epoch} Test loss:{loss}')

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(f'{save_filename}/best.pt', 'wb') as f:
                torch.save(model.state_dict(), f)
        with open(f'{save_filename}/model_{epoch + 1}.pt', 'wb') as f:
            torch.save(model.state_dict(), f)

def arguments():
    parser = argparse.ArgumentParser(description='VQ-VAE')
    # General
    parser.add_argument('--data-folder', default='./mnist/',type=str, help='name of the data folder')
    parser.add_argument('--dataset', default='mnist', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    # Latent space
    parser.add_argument('--hidden-size', type=int, default=16, help='size of the latent vectors (default: 16)')
    parser.add_argument('--k', type=int, default=512, help='number of latent vectors (default: 512)')
    # Optimization
    parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0, help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae', help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=2, help=f'number of workers for trajectories sampling (default: 2)')
    parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cpu)')

    return parser.parse_args()

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp #mp.cpu_count() - 1
    
    args = arguments()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        xx = os.environ['SLURM_JOB_ID']
        args.output_folder += f'-{xx}'
    if not os.path.exists(f'./models/{args.output_folder}'):
        os.makedirs(f'./models/{args.output_folder}')
    args.steps = 0

    main(args)
