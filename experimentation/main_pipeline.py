import spacy
import torch
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.neighbors import NearestNeighbors
import json
from tqdm import tqdm

MODEL_NAME = "gpt2"
LIMIT_CHARS = 10000
DEVICE = "mps"

def compute_twonn_id(data):
    """
    Calcule la dimension intrinsèque en utilisant l'algorithme TWO-NN.
    data: (N, D) - N points de dimension D
    """
    if len(data) < 10:
        return 0.0
    
    # Trouver les 2 plus proches voisins
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # r1 et r2 sont les distances au 1er et 2ème plus proche voisin
    # On évite les divisions par zéro
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Filtrer les cas où r1 ou r2 sont nuls ou égaux
    mu = r2 / (r1 + 1e-10)
    mu = mu[mu > 1.0]
    
    if len(mu) < 10:
        return 0.0
        
    # Estimation de la dimension via la pente du Log-Log plot
    # F(mu) = 1 - mu^(-d) => log(1 - F(mu)) = -d * log(mu)
    n = len(mu)
    mu_sorted = np.sort(mu)
    f_empirical = np.arange(1, n + 1) / n
    
    # On prend log(mu) et -log(1 - F_emp)
    x = np.log(mu_sorted)
    y = -np.log(1 - f_empirical + 1e-10)
    
    # Régression linéaire pour trouver d
    d, _ = np.polyfit(x, y, 1)
    return d

def run_full_pipeline():
    print(f"Chargement de {MODEL_NAME} sur {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
    model.eval()

    nlp = spacy.load("en_core_web_sm")

    print(f"Chargement du corpus (WikiText)...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='test', streaming=True)
    
    raw_text = ""
    for entry in dataset:
        if len(raw_text) > LIMIT_CHARS:
            break
        raw_text += entry['text'] + "\n"

    paragraphs_raw = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
    
    final_data = []
    all_embeddings_per_layer = [[] for _ in range(13)] # 13 couches (0 à 12)

    print(f"Traitement de {len(paragraphs_raw)} paragraphes...")

    for p_idx, p_text in enumerate(tqdm(paragraphs_raw)):
        doc = nlp(p_text)
        
        for s_idx, sent in enumerate(doc.sents):
            sent_text = sent.text
            # Tokenisation GPT-2
            inputs = tokenizer(sent_text, return_tensors="pt").to(DEVICE)
            gpt2_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            hidden_states = [layer.squeeze(0).cpu() for layer in outputs.hidden_states]

            gpt2_ptr = 0
            for t_idx, spacy_token in enumerate(sent):
                if spacy_token.is_space: continue
                
                word = spacy_token.text
                subword_indices = []
                reconstructed = ""

                while gpt2_ptr < len(gpt2_tokens):
                    token = gpt2_tokens[gpt2_ptr]
                    clean_token = token.replace('Ġ', '').replace('Ċ', '\n')
                    
                    reconstructed += clean_token
                    subword_indices.append(gpt2_ptr)
                    gpt2_ptr += 1
                    
                    if reconstructed.lower() == word.lower() or word.lower() in reconstructed.lower():
                        break

                if not subword_indices: continue

                # Extraction et agrégation (Moyenne) des embeddings pour ce mot
                word_embeddings = []
                for layer_idx in range(13):
                    layer_data = hidden_states[layer_idx] 
                    avg_emb = layer_data[subword_indices].mean(dim=0)
                    word_embeddings.append(avg_emb)
                    
                    all_embeddings_per_layer[layer_idx].append(avg_emb.numpy())

                final_data.append({
                    "text": word,
                    "pos": spacy_token.pos_,
                    "para_idx": p_idx,
                    "sent_idx": s_idx,
                    "pos_in_sent": t_idx,
                })

    print("\n--- Calcul de la Dimension Intrinsèque (ID) par couche ---")
    results_id = {}
    for layer_idx in range(13):
        layer_matrix = np.stack(all_embeddings_per_layer[layer_idx])
        id_val = compute_twonn_id(layer_matrix)
        results_id[f"layer_{layer_idx}"] = id_val
        print(f"Couche {layer_idx:2} | ID: {id_val:.4f}")

    return final_data, results_id

if __name__ == "__main__":
    final_corpus, ids = run_full_pipeline()
    
    with open("results_summary.json", "w") as f:
        json.dump({"intrinsic_dimensions": ids, "token_count": len(final_corpus)}, f, indent=4)
    
    print("\n✅ Pipeline terminé. Résumé sauvegardé dans 'results_summary.json'.")
