import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP  # pip install umap-learn
from sklearn.metrics import silhouette_score
from tqdm import tqdm  # pip install tqdm
import warnings
warnings.filterwarnings("ignore")

# Caminho base (ajuste conforme necessário)
base_path = r"C:\Users\prodr\Desktop\Faculdade\projeto-5-redes-profundas-projeto-v-produtos-industriais\data\Embeddings"
embedding_types = ["hog", "lbp", "bnn"]
results = []

def process_embedding(product_path, product_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    silhouette_scores = {}

    for idx, emb_type in enumerate(embedding_types):
        file_path = os.path.join(product_path, f"{emb_type}.csv")
        if not os.path.isfile(file_path):
            print(f"[!] Arquivo não encontrado: {file_path}")
            continue

        df = pd.read_csv(file_path)
        if 'label' not in df.columns:
            print(f"[!] Coluna 'label' ausente no arquivo: {file_path}")
            continue

        df = df.sample(n=500, random_state=42) if len(df) > 500 else df

        # Remove 'filename' se existir
        if 'filename' in df.columns:
            df = df.drop(columns=['filename'])

        labels = df["label"].values
        features = df.drop(columns=["label"]).values

        # Redução com UMAP
        reducer = UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(features)

        # Plotagem
        df_plot = pd.DataFrame(reduced, columns=["umap1", "umap2"])
        df_plot["label"] = labels
        sns.scatterplot(ax=axes[idx], data=df_plot, x="umap1", y="umap2",
                        hue="label", palette="tab10", s=50, legend=False)
        axes[idx].set_title(f"{emb_type.upper()}")

        # Silhouette Score
        if len(set(labels)) > 1:
            score = silhouette_score(features, labels)
            silhouette_scores[emb_type] = score
        else:
            silhouette_scores[emb_type] = float("nan")

    plt.suptitle(f"Produto: {product_name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Salva imagem na pasta do produto
    output_img_path = os.path.join(product_path, f"{product_name}_umap_comparativo.png")
    plt.savefig(output_img_path)
    plt.close()
    print(f"🖼️ Gráfico salvo em: {output_img_path}")

    return silhouette_scores

# Lista de produtos (pastas)
product_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Processa com barra de progresso
for product_name in tqdm(product_folders, desc="🔄 Processando produtos", unit="produto"):
    product_path = os.path.join(base_path, product_name)
    scores = process_embedding(product_path, product_name)
    result_row = {"Produto": product_name}
    result_row.update({f"Silhouette_{k.upper()}": v for k, v in scores.items()})
    results.append(result_row)

# Salva CSV final
df_results = pd.DataFrame(results)
output_csv = os.path.join(base_path, "silhouette_scores.csv")
df_results.to_csv(output_csv, index=False)

print(f"\n✅ Tabela final salva em: {output_csv}")
