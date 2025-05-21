from collections import Counter
import random

def undersample_dataset(image_paths, labels):
    """
    Faz undersampling para balancear as classes no dataset.

    Parâmetros:
    - image_paths: list, caminhos das imagens.
    - labels: list, rótulos correspondentes.

    Retorna:
    - new_image_paths: list, caminhos balanceados.
    - new_labels: list, labels balanceados.
    """
    counter = Counter(labels)
    min_count = min(counter.values())  # Classe minoritária

    new_image_paths = []
    new_labels = []

    for cls in counter:
        # Indices da classe
        cls_indices = [i for i, lbl in enumerate(labels) if lbl == cls]
        cls_paths = [image_paths[i] for i in cls_indices]

        # Subamostra aleatória
        undersampled_paths = random.sample(cls_paths, k=min_count)
        undersampled_labels = [cls] * min_count

        new_image_paths.extend(undersampled_paths)
        new_labels.extend(undersampled_labels)

    return new_image_paths, new_labels
