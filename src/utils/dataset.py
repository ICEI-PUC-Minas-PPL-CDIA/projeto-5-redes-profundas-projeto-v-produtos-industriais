from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T

class CableDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        transform=None,
        class_to_idx=None,
        return_both=False
    ):
        """
        Dataset de imagens de cabos para PyTorch.

        Parâmetros:
        - image_paths: lista com os caminhos absolutos das imagens.
        - labels: lista de rótulos (inteiros) correspondentes às imagens.
        - transform: transformações a serem aplicadas nas imagens (ex: albumentations).
        - class_to_idx: dicionário opcional que mapeia nome da classe → índice. Não é usado aqui diretamente, mas pode ser útil.
        - return_both: se True, retorna também a versão original da imagem (sem augmentações), útil para visualização e debug.
        """
        # Garante que temos o mesmo número de imagens e rótulos
        assert len(image_paths) == len(labels), "As listas de imagens e labels devem ter o mesmo tamanho."

        self.image_paths = image_paths              # Caminhos das imagens
        self.labels = labels                        # Labels das imagens
        self.transform = transform                  # Transformações com Albumentations
        self.class_to_idx = class_to_idx            # (opcional)
        self.return_both = return_both              # Ativa modo com imagem original + transformada

        # Transformação básica: resize + ToTensor (sem normalização)
        self.basic_tensor = T.Compose([
            T.Resize((256, 256)),                   # Redimensiona para 256x256
            T.ToTensor()                            # Converte para tensor [C, H, W] com valores entre 0 e 1
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
        - Um par (imagem_tensor, label) por padrão.
        - Se return_both=True, retorna (imagem_original_tensor, imagem_transformada_tensor, label)
        """
        try:
            # Abre a imagem e garante que ela esteja em RGB
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except UnidentifiedImageError:
            raise RuntimeError(f"Erro ao carregar a imagem: {self.image_paths[idx]}")

        label = self.labels[idx]

        # Cria a versão original (sem augmentação, apenas resize + ToTensor)
        original_tensor = self.basic_tensor(image)

        # Aplica a transformação se fornecida (ex: Albumentations)
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = original_tensor

        # Retorna os tensores conforme o modo de operação
        if self.return_both:
            return original_tensor, image_tensor, label
        else:
            return image_tensor, label
