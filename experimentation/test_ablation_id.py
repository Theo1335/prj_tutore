import spacy
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

MODEL_NAME = "gpt2"
LIMIT_CHARS = 30000 
DEVICE = "mps"

def compute_twonn_id(data):
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


def run_ablation_experiment():
    print(f"Chargement sur {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
    model.eval()

    print("Chargement spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Chargement  corpus (WikiText)...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='test', streaming=True)
    
    raw_text = ""
    for entry in dataset:
        if len(raw_text) > LIMIT_CHARS:
            break
        raw_text += entry['text'] + "\n"

    paragraphs_raw = [p.strip() for p in raw_text.split('\n\n') if p.strip()]

    conditions = ["Texte Original", "Sans NOUN", "Sans VERB", "Sans ADJ", "Sans ADP"]
    pos_to_remove_map = {
        "Texte Original": None,
        "Sans NOUN": "NOUN",
        "Sans VERB": "VERB",
        "Sans ADJ": "ADJ",
        "Sans ADP": "ADP" # Prépositions 
    }

    embeddings_per_condition = {cond: [] for cond in conditions}

    print(f"Lancement de l'expérience sur {len(paragraphs_raw)} paragraphes...")

    for p_text in tqdm(paragraphs_raw):
        doc = nlp(p_text)
        
        for sent in doc.sents:
            for condition in conditions:
                pos_target = pos_to_remove_map[condition]
 
                ablated_tokens = []
                for token in sent:
                    if pos_target is None or token.pos_ != pos_target:
                        ablated_tokens.append(token.text_with_ws)
                
                ablated_sentence = "".join(ablated_tokens).strip()
                
                if not ablated_sentence:
                    continue

                inputs = tokenizer(ablated_sentence, return_tensors="pt").to(DEVICE)
                if inputs["input_ids"].shape[1] == 0:
                    continue
                    
                with torch.no_grad():
                    outputs = model(**inputs)
                
                last_layer = outputs.hidden_states[-1].squeeze(0).cpu()
                sentence_embedding = last_layer.mean(dim=0).numpy()
                
                embeddings_per_condition[condition].append(sentence_embedding)

    results_id = {}
    for condition in conditions:
        matrix = np.stack(embeddings_per_condition[condition])
        id_val = compute_twonn_id(matrix)
        results_id[condition] = id_val
        print(f"Condition: {condition:<15} | Phrases: {len(matrix):<4} | ID Global: {id_val:.4f}")

    return results_id

def main():
    results = run_ablation_experiment()

    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), round(yval, 4), ha='center', va='bottom', fontweight='bold')

    plt.title("Expérience d'Ablation : Impact de la suppression d'une classe grammaticale\nsur la Dimension Intrinsèque (Couche 12)", fontsize=13)
    plt.ylabel("Dimension Intrinsèque Globale (ID)", fontsize=11)
    plt.ylim(0, max(values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    output_file = "ablation_id_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Ajout : création et sauvegarde d'un DataFrame pandas
    import pandas as pd
    df = pd.DataFrame({
        "Condition": labels,
        "ID": values
    })
    print(df)
    df.to_csv("ablation_id_results.csv", index=False)

if __name__ == '__main__':
    main()