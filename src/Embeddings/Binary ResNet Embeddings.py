# Versão corrigida do código original com melhorias para qualidade dos embeddings da Binary ResNet

import sys
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd

# Caminho até a pasta onde está o wrn_mcdonnell.py
sys.path.append(os.path.abspath("../binary_wide_resnet"))
from wrn_mcdonnell import WRN_McDonnell  # Arquitetura da rede binária

# Caminho base onde estão os produtos (ex: data/cable, data/screw, etc.)
BASE_DIR = "C:\\Users\\prodr\\Desktop\\Faculdade\\projeto-5-redes-profundas-projeto-v-produtos-industriais\\data"
EMBEDDINGS_BASE_DIR = os.path.join(BASE_DIR, "Embeddings")

# Inicializa o modelo binarizado
model = WRN_McDonnell(
    depth=20,
    width=10,
    num_classes=100,
    binarize=True
)
model.eval()

# Wrapper para extrair embeddings mais profundos (pós-bn + avgpool flatten)
class EmbeddingNet(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        h = F.conv2d(x, self.model.conv0, padding=1)
        h = self.model.group0(h)
        h = self.model.group1(h)
        h = self.model.group2(h)
        h = F.relu(self.model.bn(h))
        h = F.avg_pool2d(h, h.shape[-1])  # global average pooling
        return h.view(h.size(0), -1)      # flatten

embedding_model = EmbeddingNet(model)

# Transforma imagens com resolução maior e normalização RGB
transform = transforms.Compose([
    transforms.Resize((96, 96)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB
])

# Loop por todas as classes/testes para extrair os embeddings
for produto in os.listdir(BASE_DIR):
    produto_path = os.path.join(BASE_DIR, produto)
    test_dir = os.path.join(produto_path, "test")

    if not os.path.isdir(test_dir):
        continue

    print(f"\n Processando produto: {produto}")
    all_data = []

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Classe: {class_name}")
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0)

                with torch.no_grad():
                    embedding = embedding_model(image)
                    embedding = embedding.squeeze().numpy()

                all_data.append(list(embedding) + [class_name])
                print(f"{img_file} ok")

            except Exception as e:
                print(f"Erro ao processar {img_file}: {e}")

    # Salva CSV dos embeddings
    if all_data:
        produto_output_dir = os.path.join(EMBEDDINGS_BASE_DIR, produto)
        os.makedirs(produto_output_dir, exist_ok=True)
        output_csv = os.path.join(produto_output_dir, "bnn.csv")

        columns = [f"f{i}" for i in range(len(all_data[0]) - 1)] + ["label"]
        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Embeddings salvos em: {output_csv}")
    else:
        print(f"Nenhuma imagem encontrada para {produto}")

print("\n Embeddings extraídos para todos os produtos!")
