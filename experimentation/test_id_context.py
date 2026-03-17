import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.neighbors import NearestNeighbors

# --- FONCTION DE CALCUL DE LA DIMENSION INTRINSÈQUE (TWO-NN) ---
def compute_twonn_id(data):
    # L'algorithme TWO-NN nécessite un nombre minimum de points
    if len(data) < 10:
        return 0.0
    
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    mu = r2 / (r1 + 1e-10)
    mu = mu[mu > 1.0]
    
    if len(mu) < 10:
        return 0.0
        
    n = len(mu)
    mu_sorted = np.sort(mu)
    f_empirical = np.arange(1, n + 1) / n
    
    x = np.log(mu_sorted)
    y = -np.log(1 - f_empirical + 1e-10)
    
    d, _ = np.polyfit(x, y, 1)
    return d

def extract_and_compute_id(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    layer_ids = []
    # On itère sur les 13 couches (0 = Embedding, 1 à 12 = Transformers)
    for layer_tensor in outputs.hidden_states:
        # layer_tensor est de taille (1, seq_len, 768)
        # On le réduit à (seq_len, 768) pour avoir tous les tokens
        vectors = layer_tensor.squeeze(0).cpu().numpy()
        layer_id = compute_twonn_id(vectors)
        layer_ids.append(layer_id)
        
    return layer_ids

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Chargement du modèle sur {device}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()

    # On multiplie les phrases par 10 pour générer assez de points/tokens 
    # pour que le calcul géométrique de TWO-NN soit fiable (il faut une "masse" de points)
    text_sens = "The scientist discovered a new cure for the disease. I love apple juice, i drink it every morning. " * 10
    text_nimp = "Table sky running green idea desk fast Car apple roof dvd light chair gutter syringe asphalt " * 10
    text_chaos = "ijidjfirjzi fjijfi iedie ikdqiq kzdjfk zxcvbn asdfghjkl qwertyuiop " * 10
    
    print("Calcul pour le texte logique (Syntaxe & Sens)...")
    id_sens = extract_and_compute_id(text_sens, model, tokenizer, device)
    
    print("Calcul pour le texte aléatoire (Sans Syntaxe)...")
    id_nimp = extract_and_compute_id(text_nimp, model, tokenizer, device)
    
    print("Calcul pour le texte chaotique (Bruit)...")
    id_chaos = extract_and_compute_id(text_chaos, model, tokenizer, device)

    print("\nRésultats de la Dimension Intrinsèque (ID) par Couche :")
    for i in range(13):
        print(f"Couche {i:2d} | ID Logique: {id_sens[i]:.4f} | ID Aléatoire: {id_nimp[i]:.4f} | ID Chaos: {id_chaos[i]:.4f}")
    

    print("\nGénération du graphique...")
    plt.figure(figsize=(10, 6))
    layers = range(13)
    
    # Tracé des courbes
    plt.plot(layers, id_sens, marker='o', label="Texte Logique", linewidth=2, color="#2ca02c")
    plt.plot(layers, id_nimp, marker='s', label="Mots aléatoires", linewidth=2, color="#ff7f0e")
    plt.plot(layers, id_chaos, marker='^', label="Chaos (Lettres au hasard)", linewidth=2, color="#d62728")
    
    plt.title("Impact du Contexte sur la Dimension Intrinsèque (GPT-2)")
    plt.xlabel("Couche du Modèle GPT-2 (0 = Entrée brute, 12 = Sortie finale)")
    plt.ylabel("Dimension Intrinsèque (ID)")
    plt.xticks(layers)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = "id_context_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé sous : {output_file}")
    
if __name__ == '__main__':
    main()