from argparse import ArgumentParser
import os
from comet_ml import Experiment

import gym
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from models.vae_models import VAE


class GMMDataset(Dataset):
    def __init__(
        self, filename, train_or_test="train", test_prop=0.0):
        self.data = np.load('data/'+filename+'.npy', allow_pickle=True)
        n_train = int(self.data.shape[0] * (1 - test_prop))
        if train_or_test == "train":
            self.data = self.data[:n_train]
        elif train_or_test == "test":
            self.data = self.data[n_train:]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        return x


def train_vae(args):
    if 'gmm' in args.dataset:
        torch_data_train = GMMDataset(args.dataset, train_or_test='train', test_prop=args.test_split)
        dataload_train = DataLoader(torch_data_train, batch_size=args.batch_size, shuffle=True)
        torch_data_test = GMMDataset(args.dataset, train_or_test='test', test_prop=args.test_split)
        dataload_test = DataLoader(torch_data_test, batch_size=args.batch_size, shuffle=True)
        x_dim = 2

    model = VAE(x_dim=x_dim, z_dim=args.z_dim, use_fourier_features=args.use_fourier_features, n_fourier_features=args.n_fourier_features, encoder_type=args.encoder_type, decoder_type=args.decoder_type, beta=args.beta).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_test_loss = 10000000

    for ep in tqdm(range(args.n_epoch), desc="Epoch"):
        model.train()
        train_loss = 0.0
        train_kl_loss = 0.0
        train_x_loss = 0.0

        for x in dataload_train:
            x = x.type(torch.FloatTensor).cuda()
            total_loss, kl_loss, x_loss = model.get_losses(x)
            #print(kl_loss, x_loss)
            train_loss += total_loss.cpu().detach().item()
            train_kl_loss += kl_loss.cpu().detach().item()
            train_x_loss += x_loss.cpu().detach().item()
            optim.zero_grad()
            if(ep<0):
                x_loss.backward()
            else:
                total_loss.backward()
            optim.step()
        train_loss = train_loss*args.batch_size/len(torch_data_train)

        print('TRAIN LOSS: ',train_loss,train_kl_loss*args.batch_size/len(torch_data_train),train_x_loss*args.batch_size/len(torch_data_train))

        model.eval()
        test_loss = 0.0
        for x in dataload_test:
            x = x.type(torch.FloatTensor).cuda()
            total_loss, kl_loss, x_loss = model.get_losses(x)
            test_loss += total_loss.cpu().item()
        test_loss = test_loss*args.batch_size/len(torch_data_test)

        print('TEST LOSS: ',test_loss)
        if(test_loss<best_test_loss):
            best_test_loss = test_loss
            torch.save(model, 'checkpoints/'+args.dataset+'.pt')
        print('BEST TEST LOSS: ',best_test_loss)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='gmm_k_10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_split', type=float, default=0.2)

    parser.add_argument('--use_fourier_features', type=int, default=0)
    parser.add_argument('--n_fourier_features', type=int, default=128)

    parser.add_argument('--z_dim', type=int, default=1)
    parser.add_argument('--encoder_type', type=str, default='feedforward')
    parser.add_argument('--decoder_type', type=str, default='feedforward')
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()

    train_vae(args)