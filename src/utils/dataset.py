from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CableDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        """
        Dataset de imagens de cabos para PyTorch.

        Parâmetros:
        - image_paths: lista com os caminhos absolutos das imagens.
        - labels: lista de rótulos (inteiros) correspondentes às imagens.
        - transform: transformações a serem aplicadas nas imagens (ex: albumentations).
        - class_to_idx: dicionário opcional que mapeia nome da classe → índice. Não é usado aqui diretamente, mas pode ser útil.
        """
        # Garante que temos o mesmo número de imagens e rótulos
        assert len(image_paths) == len(labels), "As listas de imagens e labels devem ter o mesmo tamanho."

        self.image_paths = image_paths  # Caminhos das imagens
        self.labels = labels            # Labels das imagens
        self.transform = transform      # Transformações com Albumentations
        self.class_to_idx = class_to_idx  # (opcional)

        # Transformação básica caso nenhuma seja fornecida: resize + ToTensor
        self.basic_tensor = T.Compose([
            T.Resize((256, 256)),  # Redimensiona para 256x256
            T.ToTensor()           # Converte para tensor [C, H, W] com valores entre 0 e 1
        ])

    def __len__(self):
        # Retorna o número total de imagens no dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Recupera a imagem e o rótulo na posição idx.

        Parâmetros:
        - idx: índice da amostra no dataset.

        Retorna:
        - Um par (imagem_tensor, label)
        """
        # Abre a imagem e garante que ela esteja em RGB
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Obtém o rótulo correspondente
        label = self.labels[idx]

        # Aplica a transformação básica se não houver transformação fornecida
        original_tensor = self.basic_tensor(image)

        # Se tiver uma transformação definida, aplica ela; senão, usa a básica
        if self.transform:
            image_tensor = self.transform(image)  # usa Albumentations
        else:
            image_tensor = original_tensor        # usa Resize + ToTensor

        return image_tensor, label
