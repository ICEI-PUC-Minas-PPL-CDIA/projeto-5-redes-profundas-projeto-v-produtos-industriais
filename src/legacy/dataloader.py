import os
import math
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.mode = mode
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.basic_tensor = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ]) # Transformação simples de resize. Deve-se considerar que a imagem irá perder resolução

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
        
        original_tensor = self.basic_tensor(image)
        
        if self.transform and self.mode == "train":
            transform_tensor = self.transform(image)
            return original_tensor, transform_tensor, label
        
        else:
            image_tensor = self.transform(image) if self.transform else original_tensor
            return image_tensor,label
        
class AlbumentationsTransform:
    """
        Classe para realizar as transformações na imagem através da biblioteca Albumentations. Em cada chamada espera-se que ocorra uma modificação da imagem de treinamento, dentro de um intervalo específico de valores.
        As imagens serão modificadas levemente para que a rede possa generalizar o aprendizado. Isso tornará o modelo mais robusto.
    """
    def __init__(self, mode="train"):
        if mode == "train":
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.MotionBlur(blur_limit=(3, 7), p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.HueSaturationValue(p=0.3),
                ], p=0.7),
                A.Normalize( #Para o conjunto de dados CIFAR-100, encontrou-se (https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151) que os parâmetros para normalização da imagem são os seguintes
                            mean=(0.5071, 0.4865, 0.4409),
                            std=(0.2675, 0.2565, 0.2761)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.5071, 0.4865, 0.4409),
                            std=(0.2675, 0.2565, 0.2761)),
                ToTensorV2()
            ])

    def __call__(self, image):
        image_np = np.array(image) # de PIL para NumPy
        augmented = self.transform(image=image_np)
        return augmented['image']


def show_batch(originais, transformadas, labels):
    """
    Exibe um batch de imagens com seus respectivos rótulos.

    Parâmetros:
    - imgs: Tensor de imagens com shape [batch_size, C, H, W]
    - labels: Tensor de rótulos com shape [batch_size]

    O PyTorch trabalha com imagens no formato [C, H, W],
    mas o matplotlib espera [H, W, C].
    Por isso, fazemos o permute.
    """
    batch_size = len(labels)
    total_imgs = batch_size * 2  # original + transformada para cada item

    def get_subplot_grid(n):
        """Retorna (nrows, ncols) para n subplots com layout mais retangular possível."""
        ncols = math.floor(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        return nrows, ncols

    nrows, ncols = get_subplot_grid(total_imgs)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))
    ax = ax.flatten()

    for i in range(batch_size):
        # Imagem original
        img_o = originais[i].permute(1, 2, 0).cpu().numpy()
        img_o = np.clip(img_o, 0, 1)
        ax[2 * i].imshow(img_o)
        ax[2 * i].set_title(f"Original - Label: {labels[i].item()}")
        ax[2 * i].axis('off')

        # Imagem transformada
        img_t = transformadas[i].permute(1, 2, 0).cpu().numpy()
        img_t = np.clip(img_t, 0, 1)
        ax[2 * i + 1].imshow(img_t)
        ax[2 * i + 1].set_title("Transformada")
        ax[2 * i + 1].axis('off')

    # Oculta subplots extras, se houver
    for j in range(2 * batch_size, len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()

def unnormalize(img_tensor):
    """
    Reverte a normalização padrão aplicada com mean e std.

    Parâmetros:
    - img_tensor: Tensor [C, H, W] com valores normalizados

    Retorno:
    - Tensor denormalizado [C, H, W]
    """
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2675, 0.2565, 0.2761)

    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)  # Reverte normalização: x = x * std + mean

    return img_tensor

    
def main():
    print("Olá, mundo.")
    
    # albuTransfTrain = albumentationTrain()
    # albuTransfTest = albumentationTest()

    train_dataset = CableDataset(root_dir= BASE_DIR,
                           mode="train", 
                           transform=AlbumentationsTransform(mode="train"))
    
    test_dataset = CableDataset(root_dir=BASE_DIR,
                                mode="test",
                           transform=AlbumentationsTransform(mode="test"))

    dataloader = DataLoader(train_dataset, 
                            batch_size=8, 
                            shuffle=True, 
                            num_workers=8)

    for original,transformada, label in dataloader:
        
        # imgs_orig_unnorm = torch.stack([unnormalize(img.clone()) for img in original])
        # imgs_aug_unnorm = torch.stack([unnormalize(img.clone()) for img in transformada])

        show_batch(original, transformada, label)

        break
        
        
    
if __name__ == "__main__":
    main()