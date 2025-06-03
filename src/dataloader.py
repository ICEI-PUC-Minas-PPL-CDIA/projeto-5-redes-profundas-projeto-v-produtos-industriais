import os
import math
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Caminhos base
BASE_DIR = r"C:\Projeto 5 - Redes Neurais Profundas\data\data\cable"
class CableDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): Caminho para o diretório 'data'.
            mode (str): 'train' ou 'test'.
            transform (callable, optional): As transformações que serão aplicadas nas imagens.
            
            
            
            Deve-se colocar todas as classes na base de treinamento. Ela deve aprender quais são os defeitos. 
            Entao deve-se passar o caminho para o conjunto de teste com os defeitos. 
            No caso do balanceamento, talvez seja interessante fazer uma undersampling manual. As imagens boas estão com as umas diferenças na rotação
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        if mode == 'train':
            # Percorre subpastas (classes de defeito ou boa) dentro da pasta raiz
            dir_treino = os.path.join(root_dir,"train")
            for idx, class_name in enumerate(sorted(os.listdir(dir_treino))):
                # Criando caminho para a pasta das classes
                class_path = os.path.join(dir_treino, class_name)
                # Checando se é diretório
                if os.path.isdir(class_path):
                    #Mapeando as classes de defeito ou boa em dicionário
                    # {"bent_wire":0, "cable_swap":1, ...}
                    self.class_to_idx[class_name] = idx

                    # Percorre as imagens dentro da pasta da classe
                    for file_name in os.listdir(class_path):
                        #Checando se o nome dos arquivos terminam com png
                        if file_name.endswith(".png"):
                            #Criando o caminho da imagem
                            file_path = os.path.join(class_path, file_name)
                            #Adicionando o caminho da imagem no atributo da classe
                            self.image_paths.append(file_path)
                            #Adicionando o label da imagem no atributo da classe
                            self.labels.append(idx)
                            

        elif mode == 'test':
            dir_test = os.path.join(root_dir, 'test')
            for idx, class_name in enumerate(sorted(os.listdir(dir_test))):
                class_path = os.path.join(dir_test, class_name)
                if os.path.isdir(class_path):
                    self.class_to_idx[class_name] = idx
                for file_name in os.listdir(class_path):
                        if file_name.endswith(".png"):
                            file_path = os.path.join(class_path, file_name)
                            self.image_paths.append(file_path)
                            self.labels.append(idx)



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        """
            Deve-se diferenciar o modo de treino e de teste.
            Esse método é chamado pelo Dataloader. No caso de teste apenas transformação do resize.
            Realizar um teste no getitem para sanidade (ver se esta tudo funcionando)
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label
    

class albumentationTrain:
    """
        Classe para realizar as transformações na imagem através da biblioteca Albumentations. Em cada chamada espera-se que ocorra uma modificação da imagem de treinamento, dentro de um intervalo específico de valores.
        As imagens serão modificadas levemente para que a rede possa generalizar o aprendizado. Isso tornará o modelo mais robusto.
    """
    def __init__(self):
        
        self.transform =A.Compose([
        A.Resize(256, 256),
        A.OneOf([
            A.GaussianBlur(blur_limit=(0,10),p=0.3), #30% de chance de aplicar um GaussianBlur na imagem
            A.MotionBlur(blur_limit= (3,11),p=0.6), #60% de chance de aplicar um MotionBlur na imagem.
            A.HorizontalFlip(p=0.6),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), p=0.7),
            A.HueSaturationValue(p=0.7)
            ],
            p=0.7 #70% de chance de aplicar alguma das transformações definidas dentro do OneOf
        ),
        A.Normalize( #Para o conjunto de dados CIFAR-100, encontrou-se que os parâmetros para normalização da imagem são os seguintes
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2675, 0.2565, 0.2761)
        ),
        ToTensorV2()
        ],
        seed=10)

    def __call__(self, image):
        image = np.array(image)  # de PIL para NumPy
        augmented = self.transform(image=image)
        return augmented['image']
        
class albumentationTest:
    """
        Classe para realizar as transformações na imagem através da biblioteca Albumentations. Em cada chamada espera-se que ocorra uma modificação da imagem de treinamento, dentro de um intervalo específico de valores.
        As imagens serão modificadas levemente para que a rede possa generalizar o aprendizado. Isso tornará o modelo mais robusto.
    """
    
    def __init__(self):
        
        self.transform =A.Compose([
        A.Resize(256, 256),
        A.Normalize( #Para o conjunto de dados CIFAR-100, encontrou-se (https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151) que os parâmetros para normalização da imagem são os seguintes
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2675, 0.2565, 0.2761)
        ),
        ToTensorV2()
        ],
        seed=10)

    def __call__(self, image):
        image = np.array(image)  # de PIL para NumPy
        augmented = self.transform(image=image)
        return augmented['image']
        
        
def show_batch(imgs, labels):
    """
    Exibe um batch de imagens com seus respectivos rótulos.

    Parâmetros:
    - imgs: Tensor de imagens com shape [batch_size, C, H, W]
    - labels: Tensor de rótulos com shape [batch_size]

    O PyTorch trabalha com imagens no formato [C, H, W],
    mas o matplotlib espera [H, W, C].
    Por isso, fazemos o permute.
    """
    def get_subplot_grid(n):
        """Retorna (nrows, ncols) para n subplots com layout mais quadrado possível."""
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        return nrows, ncols

    grid_size = len(imgs)  # Número de imagens no batch
    num_linas, num_colunas = get_subplot_grid(grid_size)

    fig, ax = plt.subplots(nrows=num_linas, ncols=num_colunas, figsize=(20, 15))  # Cria a grade de subplots
    ax = ax.flatten()

    # Define o número real de imagens a serem mostradas
    num_imgs = min(len(imgs), len(ax))

    for i in range(num_imgs):
        img = imgs[i]
        img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)

        ax[i].imshow(img)
        ax[i].set_title(f"Label: {labels[i].item()}")
        ax[i].axis('off')

    # Desativa os subplots extras (se existirem)
    for j in range(num_imgs, len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()
    
    
def main():
    print("Olá, mundo.")
    
    albuTransfTrain = albumentationTrain()
    albuTransfTest = albumentationTest()

    train_dataset = CableDataset(root_dir= BASE_DIR,
                           mode="train", 
                           transform=albuTransfTrain)
    
    test_dataset = CableDataset(root_dir=BASE_DIR,
                                mode="test",
                                transform=albuTransfTest)

    dataloader = DataLoader(train_dataset, 
                            batch_size=8, 
                            shuffle=True, 
                            num_workers=8)

    for imgs, lbls in dataloader:
        show_batch(imgs, lbls)
        break
        
        
    
if __name__ == "__main__":
    main()