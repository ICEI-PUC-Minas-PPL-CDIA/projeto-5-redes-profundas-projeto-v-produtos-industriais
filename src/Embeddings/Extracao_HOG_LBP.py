# %% [markdown]
# 
# # Extração de Descritores Clássicos (HOG e LBP) de Imagens
# 
# Este notebook tem como objetivo extrair e visualizar dois descritores clássicos de imagem:
# 
# - **HOG (Histogram of Oriented Gradients)**: bom para capturar bordas e estrutura de objetos.
# - **LBP (Local Binary Pattern)**: útil para análise de textura.
# 

# %%

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern

# Parâmetros dos descritores
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

# Caminho para a pasta com as imagens
image_folder = "C:\\Users\\prodr\\OneDrive\\Documentos\\Faculdade\\projeto-5-redes-profundas-projeto-v-produtos-industriais\\data\\cable\\train\\good"


# %%

def process_hog(image_gray):
    # Extrai descritores HOG e imagem de visualização
    features, hog_image = hog(image_gray,
                              pixels_per_cell=HOG_PIXELS_PER_CELL,
                              cells_per_block=HOG_CELLS_PER_BLOCK,
                              visualize=True,
                              feature_vector=True)
    return features, hog_image


# %%

def process_lbp(image_gray):
    # Extrai descritores LBP e histograma
    lbp = local_binary_pattern(image_gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)
    return hist, lbp


# %%

hog_data = []
lbp_data = []

# Processa cada imagem individualmente
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png')): # Pegando todas as imagens .png
        print(f"Processando: {filename}")
        filepath = os.path.join(image_folder, filename)
        image = cv2.imread(filepath)
        image_gray = rgb2gray(image)

        # HOG
        hog_features, hog_image = process_hog(image_gray)
        hog_data.append([filename] + hog_features.tolist())

        # LBP
        lbp_hist, lbp_matrix = process_lbp(image_gray)
        lbp_data.append([filename] + lbp_hist.tolist())

        # Visualização
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.ravel()

        axs[0].imshow(image_gray, cmap='gray')
        axs[0].set_title("Original")

        axs[1].imshow(hog_image, cmap='gray')
        axs[1].set_title("HOG")

        axs[2].imshow(lbp_matrix, cmap='gray')
        axs[2].set_title("LBP")

        # Diferenças visuais
        hog_diff = np.abs(image_gray - hog_image)
        lbp_diff = np.abs(image_gray - lbp_matrix / lbp_matrix.max())

        axs[3].imshow(hog_diff, cmap='hot')
        axs[3].set_title("Diferença: Original - HOG")

        axs[4].imshow(lbp_diff, cmap='hot')
        axs[4].set_title("Diferença: Original - LBP")

        axs[5].axis('off')
        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# %%

# HOG
hog_columns = ['filename'] + [f'hog_{i}' for i in range(len(hog_data[0]) - 1)]
hog_df = pd.DataFrame(hog_data, columns=hog_columns)
hog_df.to_csv("descritores_hog.csv", index=False)

# LBP
lbp_columns = ['filename'] + [f'lbp_{i}' for i in range(len(lbp_data[0]) - 1)]
lbp_df = pd.DataFrame(lbp_data, columns=lbp_columns)
lbp_df.to_csv("descritores_lbp.csv", index=False)

print("Descritores HOG e LBP salvos separadamente.")



