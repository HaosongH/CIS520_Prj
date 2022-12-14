import os
import yaml
from tqdm import tqdm
import argparse
from pathlib import Path
import torch
from torch import optim
import torchvision
from torchvision import transforms
from models.vanilla_vae import *
from models.vanilla_vae_CelebA import *
from models.vanilla_vae_anime_faces import *
from torchsummary import summary

if __name__ == '__main__':
    # Iterate every file in the config to generate result
    directory = "config"
    for filename in os.listdir(directory):
        parser = argparse.ArgumentParser(description='Generic runner for VAE models')
        parser.add_argument('--experiment',  '-e',
                            dest="experiment",
                            help =  'path to the config file',
                            default='vanilla_vae')
        parser.add_argument('--dataset',  '-d',
                            dest="dataset",
                            help =  'dataset to use',
                            default='cifar10')

        args = parser.parse_args()
        with open(f"./config/{filename}", 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        # Check if cuda is available
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        if config["dataset"] == "cifar10":
            model =  VanillaVAE(**config).to(device)
            model.train()
            trainset = torchvision.datasets.CIFAR10('./datasets/cifar10/', transform=transforms.ToTensor(), download = True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                                    shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, transform=transforms.ToTensor())
            testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                                    shuffle=False, num_workers=2)
        if config["dataset"] == "anime":
            model = VanillaVAE_Anime(**config).to(device)
            model.train()
            image_size = 64
            anime_faces_dataset = torchvision.datasets.ImageFolder(root="./datasets/anime_faces/",
                                                        transform=transforms.Compose([
                                                            transforms.Resize(
                                                                image_size),
                                                            transforms.CenterCrop(
                                                                image_size),
                                                            transforms.ToTensor(),
                                                        ]))
            trainloader = torch.utils.data.DataLoader(
                anime_faces_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=8)

        if config["dataset"] == "CelebA":
            model =  CelebA_VAE(**config).to(device)
            model.train()
            trainset = torchvision.datasets.CelebA(root='./datasets/CelebA/', split="train",transform=transforms.ToTensor())
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True, num_workers=2)

            testset = torchvision.datasets.CelebA(root='./datasets/CelebA/', split="test", transform=transforms.ToTensor())
            testloader = torch.utils.data.DataLoader(testset, batch_size=12,
                                                    shuffle=False, num_workers=2)

        if config["dataset"] == "mnist":
            trainset = torchvision.datasets.MNIST(root='./datasets/mnist/', transform=transforms.ToTensor())
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

            testset = torchvision.datasets.MNIST(root='./datasets/mnist/', train=False, transform=transforms.ToTensor())
            testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                                    shuffle=False, num_workers=2)


        def train_loop():
            optimizer = optim.Adam(model.parameters(), lr=config['exp_params']['LR'], weight_decay=config['exp_params']['weight_decay'])
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['exp_params']['scheduler_gamma'])

            #iters, losses = [], []
            n = 0

            for epoch in range(config['trainer_params']['n_epochs']):
                overall_loss = 0.0
                for _, (imgs, _) in enumerate(tqdm(trainloader)):  # don't need labels here
                    optimizer.zero_grad()

                    imgs = imgs.to('cuda')
                    forward_out = model.forward(imgs)             # forward pass
                    loss_dict = model.loss_function(*forward_out)  # compute the total loss
                    loss = loss_dict['loss']
                    loss.backward()               # backward pass (compute parameter updates)
                    optimizer.step()              # make the updates for each parameter

                    # save the current training information
                    # iters.append(n)
                    # losses.append(float(loss)/16)             # compute *average* loss
                    overall_loss += loss
                    n += 1
                scheduler.step()
                    
                print("\tEpoch", epoch + 1, "complete!", "\tLoss: ", overall_loss)
                
            print("Finish!!")

        train_loop()

        torch.save(model.state_dict(), f"./ckpts/{filename}.ckpt")
