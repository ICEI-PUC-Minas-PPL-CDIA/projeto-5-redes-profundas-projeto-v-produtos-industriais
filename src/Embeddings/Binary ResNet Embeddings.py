#%%
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import sys
# Caminho absoluto até a pasta "binary_wide_resnet"
sys.path.append(os.path.abspath("../binary_wide_resnet"))
from wrn_mcdonnell import wrn20_10  # Caminho correto da sua rede
#%%
# ======== CONFIGURAÇÕES ========
DATASET_DIR = "..\\..\\data\\cable\\test" # Caminho com subpastas de imagens (ex: good/, missing_cable/)
OUTPUT_DIR = "..\\..\\data\\Embeddings"          # Pasta para salvar os CSVs por classe
FINAL_CSV = "embeddings_final.csv" # Caminho do CSV final unificado
os.makedirs(OUTPUT_DIR, exist_ok=True)
#%%
# ======== INICIALIZA A REDE BINÁRIA PRÉ-TREINADA ========
model = wrn20_10()#(num_classes=100)
model.eval()
#%%
# ======== REMOVE A ÚLTIMA CAMADA PARA GERAR EMBEDDINGS ========
class EmbeddingNet(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = torch.nn.Sequential(*list(base_model.children())[:-2])  # Remove conv_final e bn_final

    def forward(self, x):
        x = self.base(x)
        return x.view(x.size(0), -1)

embedding_model = EmbeddingNet(model)
#%%
# ======== TRANSFORMAÇÃO DA IMAGEM ========
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Tamanho padrão para CIFAR
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
#%%
# ======== PROCESSA CADA SUBPASTA DO DATASET ========
all_dfs = []

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
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                embedding = embedding_model(image)
                embedding = embedding.squeeze().numpy()

            data.append(list(embedding) + [class_name])
            print(f"✅ {img_file} ok")
        except Exception as e:
            print(f"⚠️ Erro em {img_file}: {e}")
#%%
    # SALVA O CSV DESTA CLASSE
    if data:
        columns = [f"f{i}" for i in range(len(data[0])-1)] + ["label"]
        df = pd.DataFrame(data, columns=columns)
        output_csv = os.path.join(OUTPUT_DIR, f"{class_name}.csv")
        df.to_csv(output_csv, index=False)
        print(f"💾 CSV salvo: {output_csv}")
        all_dfs.append(df)
#%%
# ======== CONCATENAÇÃO FINAL ========
if all_dfs:
    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final.to_csv(FINAL_CSV, index=False)
    print(f"📦 CSV final salvo: {FINAL_CSV}")

print("🏁 Pipeline completo!")