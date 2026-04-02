import spacy
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from tqdm import tqdm

MODEL_NAME = "gpt2"
LIMIT_CHARS = 50000
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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


def run_pos_analysis():
    print(f"Chargement sur {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
    model.eval()

    print("Chargement spaCy")
    nlp = spacy.load("en_core_web_sm")

    print("Chargement  corpus (WikiText)...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='test', streaming=True)
    
    raw_text = ""
    for entry in dataset:
        if len(raw_text) > LIMIT_CHARS:
            break
        raw_text += entry['text'] + "\n"

    paragraphs_raw = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
    
    pos_embeddings = defaultdict(list)

    print(f"Traitement de {len(paragraphs_raw)} paragraphes...")

    for p_text in tqdm(paragraphs_raw):
        doc = nlp(p_text)
        
        for sent in doc.sents:
            sent_text = sent.text
            if not sent_text.strip(): continue

            inputs = tokenizer(sent_text, return_tensors="pt").to(DEVICE)

            if inputs["input_ids"].shape[1] == 0: continue
            
            gpt2_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            last_layer_states = outputs.hidden_states[-1].squeeze(0).cpu()

            gpt2_ptr = 0
            for spacy_token in sent:
                if spacy_token.is_space or spacy_token.is_punct: 
                    continue
                
                word = spacy_token.text
                pos_tag = spacy_token.pos_
                
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

                if not subword_indices: 
                    continue

                avg_emb = last_layer_states[subword_indices].mean(dim=0).numpy()
                
                pos_embeddings[pos_tag].append(avg_emb)

    return pos_embeddings

def main():
    pos_embeddings = run_pos_analysis()
    
    MIN_SAMPLES = 50
    valid_pos_data = {}
    valid_pos_counts = {}
    
    for pos_tag, embeddings_list in pos_embeddings.items():
        if len(embeddings_list) >= MIN_SAMPLES:
            matrix = np.stack(embeddings_list)
            id_val = compute_twonn_id(matrix)
            valid_pos_data[pos_tag] = id_val
            valid_pos_counts[pos_tag] = len(embeddings_list)
            print(f"POS: {pos_tag:<5} | Mots analysés: {len(embeddings_list):<4} | ID: {id_val:.4f}")
        else:
            print(f"POS: {pos_tag:<5} | Mots analysés: {len(embeddings_list):<4} | (Ignoré, pas assez de données)")


    sorted_pos = sorted(valid_pos_data.items(), key=lambda item: item[1])
    labels = [f"{item[0]}\n(n={valid_pos_counts[item[0]]})" for item in sorted_pos]
    values = [item[1] for item in sorted_pos]

    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(labels, values, edgecolor='black')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Dimension Intrinsèque des Représentations (Couche 12) selon la Classe Grammaticale (POS)", fontsize=14)
    plt.xlabel("Part of Speech (POS Tag) & Nombre d'occurrences", fontsize=12)
    plt.ylabel("Dimension Intrinsèque (ID)", fontsize=12)
    
    plt.ylim(0, max(values) + 1.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_file = "pos_id_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()