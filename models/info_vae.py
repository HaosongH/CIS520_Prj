import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms

class InfoVAE(nn.Module):
    def __init__(self,
                latent_var = 2.,
                 **kwargs):
        super().__init__()

        self.in_channels = kwargs['model_params']['in_channels']
        self.latent_dim = kwargs['model_params']['latent_dim']
        self.hidden_dims = kwargs['model_params']['hidden_dims']

        self.kld_weight = kwargs['exp_params']['kld_weight']
        self.recon_weight = kwargs['exp_params']['recon_weight']
        self.reg_weight = kwargs['exp_params']['reg_weight']
        self.kernel_type = kwargs['exp_params']['kernel_type']
        self.alpha = kwargs['exp_params']['alpha']
        
        self.z_var = latent_var

        modules = []
        # This is the number of channels in the convolution layers, structure can be changed
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            self.in_channels = h_dim 


        self.encoder = nn.Sequential(*modules)
        # final image is 2*2*hidden_dims[-1] channels, so we multiply by 4 here
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*4, self.latent_dim) 
        self.fc_var = nn.Linear(self.hidden_dims[-1]*4, self.latent_dim)
        

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * 4)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
    
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        return  [self.decode(z), input, mu, log_var, z.detach()]

    def loss_function(self,
                      *args,
                      **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]

        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        

        recons_loss =F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = self.recon_weight * recons_loss + \
               (1. - self.alpha) * self.kld_weight * kld_loss + \
               (self.alpha + self.reg_weight - 1.)/bias_corr * mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'MMD': mmd_loss, 'KLD':kld_loss}

    def compute_kernel(self, x1, x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result


    def compute_rbf(self, x1, x2, eps = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1, x2, eps = 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd