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
BASE_DIR = r"C:\Projeto 5 - Redes Neurais Profundas\data\data\cable"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")



class CableDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): Caminho para o diretório 'data'.
            mode (str): 'train' ou 'test'.
            transform (callable, optional): As transformações que serão aplicadas nas imagens.
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if mode == 'train':
            class_dir = os.path.join(root_dir, 'train', 'good')
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(0)  # 0 para 'good'

        elif mode == 'test':
            test_dir = os.path.join(root_dir, 'test')
            defect_types = os.listdir(test_dir)
            for defect_type in defect_types:
                class_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(class_dir):
                    continue
                for img_file in os.listdir(class_dir):
                    if img_file.endswith('.png'):
                        self.image_paths.append(os.path.join(class_dir, img_file))
                        self.labels.append(0 if defect_type == 'good' else 1)  # 1 para defeito

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = rgb2gray(cv2.imread(self.image_paths[idx]))
        
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
    


def main():
    print("Olá, mundo.")
    
    
    
if __name__ == "__main__":
    main()