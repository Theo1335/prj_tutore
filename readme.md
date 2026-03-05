# Exploration du Langage et du Cerveau via l'Intelligence Artificielle

Ce dépôt contient les travaux de recherche et d'implémentation du **Projet Tutoré** réalisé à l'Université Paris 8 (Licence Informatique & Vidéoludisme) pour l'année universitaire 2025-2026.

## Objectif du Projet

L'objectif est d'analyser le parallèle entre le traitement du langage par les modèles de type **Transformer** et les mécanismes cognitifs du **cerveau humain**. Nous cherchons à déterminer si l'IA traite la sémantique de manière analogue au cerveau, notamment lors de l'unification des mots pour former une phrase cohérente.

## 🧠 Fondements Théoriques

Le projet s'appuie sur trois piliers scientifiques :


* **Modèle MUC (Memory, Unification, Control)** : Un cadre neurolinguistique décrivant comment le cerveau stocke, assemble et contrôle les unités linguistiques.


* **Dimensionnalité Intrinsèque (ID)** : Une mesure de la complexité géométrique des données. Nous utilisons l'algorithme **TwoNN** pour estimer cette dimensionnalité au sein des couches du modèle.


* **Profil "Hunchback"** : L'observation d'une phase d'expansion suivie d'une compression de l'information sémantique à travers les couches d'un réseau profond.



## 🧪 Méthodologie Expérimentale

Nous comparons l'activité interne des modèles face à deux types de stimuli :

1. **Groupe Sémantique** : Phrases naturelles et cohérentes issues de Wikipedia.


2.  **Groupe Jabberwocky** : Phrases syntaxiquement correctes mais dénuées de sens (ex: *"La vlipure d'Elsa Barraine est jitrement carlouée"*).



### Modèles étudiés

*  **GPT-2 Small** : Pour son architecture causale proche de la lecture humaine.


*  **Llama 3.1 (8B)** : Pour généraliser les observations sur une architecture moderne de grande échelle.



## 📅 Roadmap du Projet

*  **Février** : Mise en place de l'environnement technique (PyTorch, Hugging Face) et création du générateur de Jabberwocky.


*  **Mars** : Extraction des *hidden states* et calcul de l'ID pour tester l'effet de **ramping** (accumulation d'information).


*  **Avril** : Analyse comparative des profils de compression et rédaction du rapport final.


## 👥 Équipe

*  **Étudiants** : Nicolas ROY & Theo MASSENYA 


*  **Tuteur** : Mme Revekka KYRIAKOGLOU 


