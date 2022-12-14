# CIS5200-Final-Proj

Directions to train VAEs on different datasets.

Use ```python run.py -e vanilla_vae -d cifar10``` 
to train a vanilla vae model on the CIFAR10 dataset

Similar for other models and datasets. Available models include vanilla_vae, info_vae. (beta_vae is just vanilla vae with KLD_Weight != 1)
Available datasets include MNIST, CIFAR10, anime, CELEBA

configuration files are saved in ```./config``` in the yaml format.

Instructions on how to mix and match VAEs are in ```mix.ipynb```, instructions on how to visualize VAE quality in ```visualize.ipynb```
