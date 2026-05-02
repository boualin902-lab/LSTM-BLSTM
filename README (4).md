# TP6 — Classification de Textes : LSTM, GRU, BGRU & BLSTM

> **Deep Learning | NLP | Keras / TensorFlow**  
> Classification de SMS spam/ham avec des architectures récurrentes.

---

## 📋 Description

Ce projet implémente un pipeline complet de classification de textes à partir de la base de données **SMS Spam Collection**. Il compare quatre architectures de réseaux de neurones récurrents :

| Modèle | Type | Description |
|--------|------|-------------|
| LSTM | Unidirectionnel | Long Short-Term Memory |
| GRU | Unidirectionnel | Gated Recurrent Unit |
| BGRU | Bidirectionnel | Bidirectional GRU |
| BLSTM | Bidirectionnel | Bidirectional LSTM |

---

## 🎯 Objectifs du TP

- Préparer des données textuelles (tokenisation, padding, encodage)
- Construire, entraîner et évaluer des modèles LSTM / GRU / BLSTM
- Analyser et interpréter l'overfitting
- Comparer les performances des différentes architectures

---

## 📁 Structure du projet

```
TP6_LSTM_BLSTM/
│
├── TP6_LSTM_BLSTM.ipynb   # Notebook Jupyter principal (code + analyses)
├── README.md              # Ce fichier
└── data/
    └── spam.csv           # Base de données SMS Spam (à télécharger)
```

---

## ⚙️ Installation

### Prérequis

- Python 3.8+
- pip

### Dépendances

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

> **Avec conda :**
> ```bash
> conda create -n tp6 python=3.10
> conda activate tp6
> pip install tensorflow scikit-learn pandas matplotlib seaborn
> ```

---

## 🗂️ Base de données

La base **SMS Spam Collection** contient 5 572 messages SMS étiquetés :
- `ham` (0) : message légitime
- `spam` (1) : message indésirable

Elle est automatiquement téléchargée dans le notebook. Vous pouvez aussi la télécharger manuellement sur [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

---

## 🚀 Utilisation

```bash
# Cloner le projet
git clone https://github.com/votre-username/TP6-LSTM-BLSTM.git
cd TP6-LSTM-BLSTM

# Lancer le notebook
jupyter notebook TP6_LSTM_BLSTM.ipynb
```

Exécuter les cellules dans l'ordre — le pipeline est entièrement automatisé.

---

## 📊 Pipeline de traitement

```
SMS bruts
    ↓
LabelEncoder (ham→0, spam→1)
    ↓
Tokenizer Keras (texte → séquences d'entiers)
    ↓
pad_sequences (uniformisation à max_len=150)
    ↓
Split Train (2/3) / Test (1/3)
    ↓
Modèles : LSTM → GRU → BGRU → BLSTM
    ↓
Évaluation & Comparaison
```

---

## 🏗️ Architectures des modèles

### LSTM (base)
```
Embedding(1000, 50, input_length=150)
→ LSTM(64, dropout=0.2)
→ Dense(256, relu)
→ Dense(1, sigmoid)
```

### BLSTM
```
Embedding(1000, 100, input_length=150)
→ Bidirectional(LSTM(64, dropout=0.2))
→ Dense(256, relu) + Dropout(0.3)
→ Dense(1, sigmoid)
```

---

## 📈 Résultats typiques

| Modèle | Accuracy Test | Epochs (arrêt) |
|--------|:-------------:|:--------------:|
| LSTM | ~97.5% | ~5–8 |
| LSTM amélioré | ~98.0% | ~6–10 |
| GRU | ~97.8% | ~5–7 |
| BGRU | ~98.2% | ~5–8 |
| BLSTM | ~98.5% | ~5–8 |

> *Les résultats varient légèrement selon l'initialisation aléatoire.*

---

## 🔍 Points clés analysés

### Overfitting
- Détecté quand `val_loss` remonte pendant que `train_loss` baisse
- Solutions appliquées : **Dropout**, **EarlyStopping** (`monitor='val_loss'`, `patience=3`)

### Padding (I.4.d)
`pad_sequences(sequences, maxlen=150)` unifie la longueur des séquences :
- Tronque les séquences trop longues (> 150 mots)
- Ajoute des zéros au début pour les séquences trop courtes

### Bidirectionnel vs Unidirectionnel
Les modèles bidirectionnels (BGRU, BLSTM) lisent la séquence dans les **deux sens**, capturant un contexte plus riche — au prix d'un coût calcul doublé.

---

## 📦 Bibliothèques utilisées

```python
tensorflow / keras    # Modèles deep learning
scikit-learn          # LabelEncoder, train_test_split
pandas / numpy        # Manipulation des données
matplotlib / seaborn  # Visualisations
```

---

## 👤 Auteur

Réalisé dans le cadre du **TP 6 — Deep Learning / NLP**  
Module : Apprentissage Automatique & Deep Learning

---

## 📄 Licence

Usage académique — libre de réutilisation pour l'enseignement.
