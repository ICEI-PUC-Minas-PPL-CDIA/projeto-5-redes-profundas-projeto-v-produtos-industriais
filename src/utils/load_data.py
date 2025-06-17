import os

def load_image_paths_and_labels(root_dir, exts=('.png')):
    """
    Carrega os caminhos das imagens e seus rótulos a partir da estrutura de diretórios.

    Parâmetros:
    - root_dir: str, caminho para a pasta raiz do dataset.
    - exts: tuple, extensões de arquivos aceitas como imagens.

    Retorna:
    - image_paths: list, caminhos completos das imagens.
    - labels: list, índice da classe de cada imagem.
    - class_to_idx: dict, mapeamento de nome da classe para índice.
    """
    
    image_paths = []
    labels = []
    
    # Lista as classes ordenadamente para manter consistência no mapeamento
    class_names = sorted([
        d for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    
    # Mapeia nome da classe para um índice
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(exts):
                path = os.path.join(class_dir, fname)
                image_paths.append(path)
                labels.append(class_to_idx[class_name])
    
    return image_paths, labels, class_to_idx
