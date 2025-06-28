from sklearn.model_selection import train_test_split

def split_dataset(image_paths, labels, test_size=0.2, random_state=42):
    """
    Divide os dados em treino e teste mantendo a proporção das classes.

    Parâmetros:
    - image_paths: list, caminhos completos das imagens.
    - labels: list, rótulos das imagens.
    - test_size: float, proporção do conjunto de teste.
    - random_state: int, semente para reprodutibilidade.

    Retorna:
    - (train_paths, train_labels), (test_paths, test_labels)
    """
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    return (train_paths, train_labels), (test_paths, test_labels)
