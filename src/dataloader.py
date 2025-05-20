import os
import cv2
import math
import torch
import torchvision
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from torch.utils.data import Dataset, DataLoader


HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'


# Caminhos base
BASE_DIR = r"C:\Projeto 5 - Redes Neurais Profundas\data\data"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")



class CableDataset(Dataset):
    
    def __init__(self, transform=None):
        ## Carregamento dos dados
        xy = np.loadtxt('./data/cable.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.X = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,0]) # A coluna 0 está com a classe da imagem neste exemplo
        self.n_samples = xy.shape[0] #número de linhas
        
        self.transform = transform

    def __getitem__(self, index):
        ## Permite a indexacao dos items
        sample = self.X[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
    def __len__(self):
        # retorna o tamanho do dataset
        return self.n_samples
    
class ToTensor:
    
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    

dataset = CableDataset(transform=ToTensor())
dataloader = DataLoader(dataset=dataset, 
                        batch_size=256,
                        shuffle=True,
                        num_workers=4)

#loop de treinamento
num_epocas = 10
total_amostras = len(dataset)
n_iterations = math.ceil(total_amostras/256) #num de iterações é o tamanho total pelo tamanho do batchsize

for epoca in range(num_epocas):
    for i, (inputs, labels) in enumerate(dataloader):
        # Realizar o forward backward, update
        if(1+1)%5 == 0:
            print(f'Época{epoca+1}/{num_epocas}, step {i+1}/{n_iterations}, inputs{inputs.shape}')

dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)