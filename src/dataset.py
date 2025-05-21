from PIL import Image
import numpy as np
from torch.utils.data import Dataset
class CableDataset(Dataset):
    """
    Classe dataset para importar as imagens dos cabos 

    Parâmetros:
    - image_paths: list, caminhos das imagens.
    - labels: list, rótulos correspondentes.
    - transform: callable, transformações aplicadas na imagem.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carrega a imagem
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)
        
        label = self.labels[idx]

        # Aplica transformações, se existirem
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label


