import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.normal as Normal
import torch.distributions.categorical as Categorical
import torch.distributions.mixture_same_family as MixtureSameFamily
import torch.distributions.kl as KL


class Encoder(nn.Module):
    '''
    Feed-Forward Encoder
    '''
    def __init__(self,x_dim,z_dim,h_dim=[256,256,64,32]):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.emb_layer = nn.Sequential(nn.Linear(x_dim,h_dim[0]),nn.ReLU(),nn.Linear(h_dim[0],h_dim[1]),nn.ReLU())
        self.mean_layer = nn.Sequential(nn.Linear(h_dim[1],h_dim[2]),nn.ReLU(),nn.Linear(h_dim[2],h_dim[3]),nn.ReLU(),nn.Linear(h_dim[3],z_dim))
        self.sig_layer = nn.Sequential(nn.Linear(h_dim[1],h_dim[2]),nn.ReLU(),nn.Linear(h_dim[2],h_dim[3]),nn.ReLU(),nn.Linear(h_dim[3],z_dim),nn.Softplus())

    
    def forward(self, x):
        x_emb = self.emb_layer(x)
        z_means = self.mean_layer(x_emb)
        z_sigs = self.sig_layer(x_emb)

        return z_means, z_sigs


class Decoder(nn.Module):
    '''
    Feed-Forward Decoder
    '''
    def __init__(self,x_dim,z_dim,h_dim=[256,256,64,32]):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.emb_layer = nn.Sequential(nn.Linear(z_dim,h_dim[0]),nn.ReLU(),nn.Linear(h_dim[0],h_dim[0]),nn.ReLU(),nn.Linear(h_dim[0],h_dim[1]),nn.ReLU())
        self.mean_layer = nn.Sequential(nn.Linear(h_dim[1],h_dim[2]),nn.ReLU(),nn.Linear(h_dim[2],h_dim[3]),nn.ReLU(),nn.Linear(h_dim[3],x_dim))
        self.sig_layer = nn.Sequential(nn.Linear(h_dim[1],h_dim[2]),nn.ReLU(),nn.Linear(h_dim[2],h_dim[3]),nn.ReLU(),nn.Linear(h_dim[3],x_dim),nn.Softplus())


    def forward(self, z):
        z_emb = self.emb_layer(z)
        x_means = self.mean_layer(z_emb)
        x_sigs = self.sig_layer(z_emb)

        return x_means, x_sigs


class FourierFeaturesLayer(nn.Module):
    def __init__(self, input_dim, feature_dim, min_power=1, max_power=4):
        super(FourierFeaturesLayer, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        # Initialize the frequency encodings
        #self.freq_encodings = nn.Parameter(torch.randn(feature_dim, input_dim))
        #self.B = torch.randn((feature_dim,input_dim)).cuda()*10
        self.B = torch.eye(input_dim)*(2**min_power)
        for i in range(min_power+1,max_power+1):
            B_i = torch.eye(input_dim)*(2**i)
            self.B = torch.vstack([self.B,B_i])
        self.B = self.B.cuda()


    def forward(self, x):
        # Reshape input to have a shape (batch_size, input_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Compute the Fourier features
        features = np.pi * (x@self.B.T)#torch.matmul(x, self.freq_encodings.t())

        # Concatenate the cosine and sine components
        features = torch.cat([torch.cos(features), torch.sin(features)], dim=-1)

        features = features.view(batch_size, -1)
        return features


class VAE(nn.Module):
    '''
    VAE Model
    '''
    def __init__(self,x_dim,z_dim,use_fourier_features=False,n_fourier_features=128,encoder_type='feedforward',decoder_type='feedforward',encoder_h_dim=[256,256,64,32],decoder_h_dim=[256,256,64,32],beta=1.0):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.beta = beta
        self.use_fourier_features = use_fourier_features
        self.n_fourier_features = n_fourier_features

        if encoder_type == 'feedforward':
            self.encoder = Encoder(x_dim=x_dim,z_dim=z_dim,h_dim=encoder_h_dim)
        if decoder_type == 'feedforward' and not use_fourier_features:
            self.decoder = Decoder(x_dim=x_dim,z_dim=z_dim,h_dim=decoder_h_dim)
        elif decoder_type == 'feedforward' and use_fourier_features:
            self.ff_layer = FourierFeaturesLayer(input_dim=z_dim,feature_dim=n_fourier_features)
            self.decoder = Decoder(x_dim=x_dim,z_dim=2*n_fourier_features+z_dim,h_dim=decoder_h_dim)


    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps


    def forward(self, x):
        z_means, z_sigs = self.encoder(x)
        z_sampled = self.reparameterize(z_means,z_sigs)
        if self.use_fourier_features:
            z_ff = self.ff_layer(z_sampled).detach()
            z_sampled = torch.cat([z_sampled, z_ff], axis=-1)
        x_means, x_sigs = self.decoder(z_sampled)

        return x_means, x_sigs, z_means, z_sigs


    def get_losses(self, x):
        x_means, x_sigs, z_means, z_sigs = self.forward(x)
        z_post_dist = Normal.Normal(z_means, z_sigs)
        z_prior_means = torch.zeros_like(z_means)
        z_prior_sigs = torch.ones_like(z_sigs)
        z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)
        x_dist = Normal.Normal(x_means, x_sigs)

        kl_loss = torch.mean(KL.kl_divergence(z_post_dist, z_prior_dist))
        x_loss = -torch.mean(x_dist.log_prob(x))
        #x_loss = torch.mean(torch.sum((x_means-x)**2,axis=1))
        total_loss = x_loss + self.beta*kl_loss
        #print(x_means[0],x[0])

        return total_loss, kl_loss, x_loss