#%%
import os
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern

#%%==== Parâmetros dos descritores
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

# Caminhos base
BASE_DIR = r"/home/pedro/Documentos/Faculdade/projeto-5-redes-profundas-projeto-v-produtos-industriais/data"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")

#%%==== Funções auxiliares
def process_hog(image_gray):
    features, _ = hog(image_gray,
                      pixels_per_cell=HOG_PIXELS_PER_CELL,
                      cells_per_block=HOG_CELLS_PER_BLOCK,
                      visualize=True,
                      feature_vector=True)
    return features

def process_lbp(image_gray):
    lbp = local_binary_pattern(image_gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)
    return hist

#%%==== Loop por todos os produtos em data/
for produto in os.listdir(BASE_DIR):
    produto_path = os.path.join(BASE_DIR, produto)
    if not os.path.isdir(produto_path):
        continue

    print(f"\n📦 Produto: {produto}")
    hog_data = []
    lbp_data = []

    # Varre cada classe diretamente dentro do produto
    for label in os.listdir(produto_path):
        class_dir = os.path.join(produto_path, label)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if not filename.lower().endswith('.png'):
                continue

            filepath = os.path.join(class_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue

            image_gray = rgb2gray(image)

            # Extrair HOG e LBP
            hog_features = process_hog(image_gray)
            lbp_features = process_lbp(image_gray)

            hog_data.append([filename, label] + hog_features.tolist())
            lbp_data.append([filename, label] + lbp_features.tolist())

        print(f"✅ Classe '{label}' processada.")

    # Criar diretório de saída por produto
    produto_output_dir = os.path.join(EMBEDDINGS_DIR, produto)
    os.makedirs(produto_output_dir, exist_ok=True)

    # Gerar colunas
    if hog_data:
        hog_columns = ['filename', 'label'] + [f'hog_{i}' for i in range(len(hog_data[0]) - 2)]
        df_hog = pd.DataFrame(hog_data, columns=hog_columns)
        df_hog.to_csv(os.path.join(produto_output_dir, "hog.csv"), index=False)

    if lbp_data:
        lbp_columns = ['filename', 'label'] + [f'lbp_{i}' for i in range(len(lbp_data[0]) - 2)]
        df_lbp = pd.DataFrame(lbp_data, columns=lbp_columns)
        df_lbp.to_csv(os.path.join(produto_output_dir, "lbp.csv"), index=False)

    print(f"💾 Embeddings HOG e LBP salvos para '{produto}'.")

print("\n🏁 Extração de HOG e LBP concluída para todos os produtos!")

# %%
