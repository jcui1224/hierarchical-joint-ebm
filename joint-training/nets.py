import torch as t
import torch.nn as nn
import numpy as np
from collections import OrderedDict
mse = nn.MSELoss(reduction='none')
import torch.nn.functional as F
# from distribution import N1 as N
class reshape(nn.Module):
    def __init__(self, *args):
        super(reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class _Cifar10_netG(nn.Module):
    def __init__(self, z1_dim, sigma):
        super().__init__()
        ngf = 64 # ngf of lebm
        self.input_z_dim = z1_dim
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(self.input_z_dim, ngf * 16, 8, 1, 0, bias=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf * 4, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        assert z.size(1) == self.input_z_dim
        z = z.view(-1, self.input_z_dim, 1, 1)
        mu = self.decode(z) # 3, 32, 32
        return mu

    def loss(self, x, z1_q):
        x_rec = self(z1_q)
        pxgz1_loss = t.sum(mse(x_rec, x), dim=[1, 2, 3]).mean()
        return pxgz1_loss

class _Cifar10_netI(nn.Module):
    def __init__(self, z1_dim, nif, N):
        super().__init__()
        self.to_z_dim = z1_dim
        self.N = N
        self.encode = nn.Sequential(
            nn.Conv2d(3, nif*4, 3, 1, 1), # 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(nif * 4, nif * 8, 4, 2, 1), # 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(nif * 8, nif * 16, 4, 2, 1), # 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(nif * 16, self.to_z_dim*2, 8, 1, 0),  # self.to_z_dim*2
        )

    def forward(self, x):
        # z = z.view(-1, self.input_z_dim, 1, 1)

        logits = self.encode(x).squeeze() # (bs, z1 dim)
        mu, log_sig = logits.chunk(2, dim=1)
        # mu = self.mu(logit)
        # log_sig = self.log_sig(logit)
        dist = self.N(mu, log_sig, device=x.device)
        z1 = dist.rsample()
        return dist, z1

class _netEzi(nn.Module):
    def __init__(self, z_dim, nef, num_layers):
        super().__init__()
        self.z_dim = z_dim
        current_dims = self.z_dim
        layers = OrderedDict()
        for i in range(num_layers):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, nef)
            layers['lrelu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = nef

        layers['out'] = nn.Linear(current_dims, 1)
        self.energy = nn.Sequential(layers)

    def forward(self, z):
        assert z.shape[1] == self.z_dim
        z = z.view(-1, self.z_dim)
        en = self.energy(z).squeeze(1)
        return en

class _netGzi_mlp(nn.Module):
    def __init__(self, input_z_dim, to_z_dim, ndf, num_layers, N):
        super().__init__()
        self.input_z_dim = input_z_dim
        self.to_z_dim = to_z_dim
        self.N = N
        current_dims = self.input_z_dim
        layers = OrderedDict()
        for i in range(num_layers):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, ndf)
            layers['lrelu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = ndf

        layers['out'] = nn.Linear(current_dims, self.to_z_dim*2)
        self.logit = nn.Sequential(layers)

    def forward(self, z):
        assert z.size(1) == self.input_z_dim

        logits = self.logit(z)
        mu, log_sig = logits.chunk(2, dim=1)
        # log_sig = self.log_sig(logit)
        dist = self.N(mu, log_sig, device=z.device)
        zi = dist.rsample()
        return dist, zi

    def loss(self, z_q, z_p):
        z1_q, z2_q = z_q[0], z_q[1]
        z1_p, z2_p = z_p[0], z_p[1]

        z1_q_dist, _ = self(z2_q)
        pz1gz2_t = t.sum(z1_q_dist.log_prob(z1_q), dim=1).mean()
        z1_p_dist, _ = self(z2_p)
        pz1gz2_f = t.sum(z1_p_dist.log_prob(z1_p), dim=1).mean()

        pz1gz2_loss = pz1gz2_f - pz1gz2_t

        return pz1gz2_loss

class _netIzi_mlp(nn.Module):
    def __init__(self, input_z_dim, to_z_dim, nif, num_layers, N):
        super().__init__()
        self.input_z_dim = input_z_dim
        self.to_z_dim = to_z_dim
        self.N = N
        current_dims = self.input_z_dim
        layers = OrderedDict()
        for i in range(num_layers):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, nif)
            layers['relu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = nif

        layers['out'] = nn.Linear(current_dims, self.to_z_dim*2)
        self.logit = nn.Sequential(layers)
        #
        # self.mu = nn.Sequential(
        #     nn.Linear(nif, self.to_z_dim)
        # )
        # self.log_sig = nn.Sequential(
        #     nn.Linear(nif, self.to_z_dim),
        # )
    def forward(self, z):
        assert z.size(1) == self.input_z_dim

        logits = self.logit(z)
        mu, log_sig = logits.chunk(2, dim=1)
        # mu = self.mu(logit)
        # log_sig = self.log_sig(logit)
        dist = self.N(mu, log_sig, device=z.device)
        zi = dist.rsample()
        return dist, zi

