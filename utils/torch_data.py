import torch
import torchvision

cifar10_data = torchvision.datasets.CIFAR10('../datasets/cifar10/', download=True)
fashion_mnist_data = torchvision.datasets.FashionMNIST('../datasets/fashion_mnist/', download=True)
mnist_data = torchvision.datasets.MNIST('../datasets/mnist/', download=True)
human_face_data = torchvision.datasets.CelebA('../datasets/CelebA/', download = True)