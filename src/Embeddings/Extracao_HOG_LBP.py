#%%
import os
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
#%%
# Parâmetros dos descritores
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
#%%
# Diretório base com as pastas de cada classe
base_dir = "..\\..\\data\\cable\\test"
#%%
hog_data = []
lbp_data = []
combined_data = []
#%%
# Funções auxiliares
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
#%%
# Loop pelas classes (subpastas)
for label in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, label)
    if not os.path.isdir(class_dir):
        continue

    for filename in os.listdir(class_dir):
        if filename.lower().endswith('.png'):
            filepath = os.path.join(class_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            image_gray = rgb2gray(image)

            # HOG
            hog_features = process_hog(image_gray)
            hog_data.append([filename, label] + hog_features.tolist())

            # LBP
            lbp_features = process_lbp(image_gray)
            lbp_data.append([filename, label] + lbp_features.tolist())

            # Combinação
            combined = np.concatenate([hog_features, lbp_features])
            combined_data.append([filename, label] + combined.tolist())
            
    print(f"Pasta '{label}' processada com sucesso. {len(os.listdir(class_dir))} arquivos lidos.")

#%%
# Salvar CSVs
hog_columns = ['filename', 'label'] + [f'hog_{i}' for i in range(len(hog_data[0]) - 2)]
lbp_columns = ['filename', 'label'] + [f'lbp_{i}' for i in range(len(lbp_data[0]) - 2)]
combined_columns = ['filename', 'label'] + [f'hog_{i}' for i in range(len(hog_data[0]) - 2)] + [f'lbp_{i}' for i in range(len(lbp_data[0]) - 2)]

pd.DataFrame(hog_data, columns=hog_columns).to_csv("hog_features.csv", index=False)
pd.DataFrame(lbp_data, columns=lbp_columns).to_csv("lbp_features.csv", index=False)
pd.DataFrame(combined_data, columns=combined_columns).to_csv("hog_lbp_combined.csv", index=False)

print("Arquivos salvos com sucesso!")
