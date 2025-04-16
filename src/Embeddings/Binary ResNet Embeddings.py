#%%
import sys
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd

# Caminho até a pasta onde está o wrn_mcdonnell.py
sys.path.append(os.path.abspath("../binary_wide_resnet"))
from wrn_mcdonnell import WRN_McDonnell  # Importa a arquitetura da rede binária

#%%====== CONFIGURAÇÕES ========
DATASET_DIR = "..\\..\\data\\cable\\test"            # Caminho com subpastas de imagens (ex: good/, missing_cable/)
OUTPUT_DIR = "..\\..\\data\\Embeddings"              # Pasta onde os CSVs por classe serão salvos
FINAL_CSV = os.path.join(OUTPUT_DIR, "embeddings_final.csv")  # Caminho do CSV final combinado
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%%======= INICIALIZA O MODELO BINÁRIO WRN-20-10 ========
model = WRN_McDonnell(
    depth=20,          # Profundidade da rede (3 blocos por grupo)
    width=10,          # Largura aumentada (wide residual network)
    num_classes=100,   # Não afeta pois a última camada será removida
    binarize=True      # Ativa pesos e ativações binarizados
)
model.eval()  # Coloca o modelo em modo de inferência

#%%======= REMOVE A ÚLTIMA CAMADA PARA USAR COMO EXTRATOR DE EMBEDDINGS ========
class EmbeddingNet(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        # Copia o forward do modelo original até antes da última camada
        h = F.conv2d(x, self.model.conv0, padding=1)
        h = self.model.group0(h)
        h = self.model.group1(h)
        h = self.model.group2(h)
        h = F.relu(self.model.bn(h))
        return h.view(h.size(0), -1)  # Flatten final

embedding_model = EmbeddingNet(model)

#%%======= TRANSFORMAÇÃO DAS IMAGENS ========
transform = transforms.Compose([
    transforms.Resize((32, 32)),              # WRN-20-10 espera imagens de 32x32
    transforms.ToTensor(),                    # Converte PIL para Tensor (C, H, W)
    transforms.Normalize((0.5,), (0.5,))      # Normaliza para a escala [-1, 1]
])

# Lista para armazenar todos os DataFrames por classe
all_dfs = []

#%%======= LOOP PRINCIPAL: VARRE CADA SUBPASTA DO DATASET ========
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"🔍 Processando classe: {class_name}")
    data = []

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)  # Adiciona dimensão de batch

            with torch.no_grad():
                embedding = embedding_model(image)
                embedding = embedding.squeeze().numpy()

            # Salva embedding + rótulo
            data.append(list(embedding) + [class_name])
            print(f"✅ {img_file} ok")

        except Exception as e:
            print(f"⚠️ Erro ao processar {img_file}: {e}")

    if data:
        columns = [f"f{i}" for i in range(len(data[0]) - 1)] + ["label"]
        df = pd.DataFrame(data, columns=columns)
        output_csv = os.path.join(OUTPUT_DIR, f"{class_name}.csv")
        df.to_csv(output_csv, index=False)
        print(f"💾 CSV salvo: {output_csv}")
        all_dfs.append(df)

#%%======= CONCATENA TODOS OS CSVs EM UM ÚNICO ARQUIVO FINAL ========
if all_dfs:
    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final.to_csv(FINAL_CSV, index=False)
    print(f"📦 CSV final salvo: {FINAL_CSV}")

print("🏁 Extração de embeddings concluída com sucesso!")


