from argparse import ArgumentParser
import os
from comet_ml import Experiment

import gym
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from models.vae_models import VAE


def collect_GMM_dataset(n_clusters=3, std=0.25, n_samples=10000, means=None):
	if means is None:
		means = np.random.uniform(-10.0, 10.0, size=(n_clusters,2))
	samples = np.zeros((n_samples,2))

	for i in range(n_samples):
		n = np.random.choice(n_clusters)
		samples[i,0] = np.random.normal(means[n,0], std, 1)
		samples[i,1] = np.random.normal(means[n,1], std, 1)

	return samples, means


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--std', type=float, default=0.25)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--means', type=float, default=None)
    args = parser.parse_args()

    samples, means = collect_GMM_dataset(args.n_clusters, args.std, args.n_samples, args.means)

    plt.scatter(samples[:,0], samples[:,1])
    plt.scatter(means[:,0], means[:,1], color='red', marker='x')
    plt.savefig('plots/gmm_k_'+str(args.n_clusters)+'.png')
    np.save('data/gmm_k_'+str(args.n_clusters)+'.npy', samples)