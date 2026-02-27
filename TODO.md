# TODO -- Plan de travail du projet

## Projet
**Classification multi-label de commentaires toxiques** (Jigsaw Toxic Comment Classification Challenge, Kaggle).
Objectif : demontrer qu'un modele NLP recent surpasse une baseline classique.

## Dataset
- **Source** : Kaggle - Jigsaw Toxic Comment Classification Challenge
- **Taille** : ~160 000 commentaires Wikipedia
- **Labels** (6, binaires, multi-label) : `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Fichiers bruts** : `train.csv`, `test.csv`, `test_labels.csv`, `sample_submission.csv`
- **Localisation** : `data/raw/`
- **Strategie de split** :
  - `train.csv` -> split en train (70%) / val (15%) / test (15%) -- stratifie, seed fixe
  - `test.csv` Kaggle -> reserve pour simulation de submission finale (non utilise pendant l'entrainement)

## Modeles a comparer

### Baselines (classiques + reference deep learning)
| Modele | Description |
|--------|-------------|
| TF-IDF + Logistic Regression | Reference classique, interpretable |
| TF-IDF + Linear SVM | Performant en haute dimension |
| TF-IDF + Multinomial Naive Bayes | Baseline probabiliste simple |
| BERT (bert-base-uncased) | Baseline deep learning de reference (2018) |

### Modeles recents (Transformers)
| Modele | Description | Ref |
|--------|-------------|-----|
| DistilBERT (distilbert-base-uncased) | BERT distille, plus leger | arXiv 2019 |
| ModernBERT (answerdotai/ModernBERT-base) | Encodeur moderne, 2T tokens, seq 8192 | arXiv Dec 2024 |
| NeoBERT (neobert-base) | Next-gen BERT, 250M params, bat ModernBERT sur MTEB | arXiv Feb 2025 |

## Metriques
- ROC-AUC (macro) -- metrique principale
- F1-score (macro)
- Precision / Recall (macro)
- Hamming Loss
- Temps d'entrainement / inference

---

## Taches

### Phase 1 -- Setup
- [ ] Creer la branche de travail depuis main
- [ ] Mettre en place la structure de dossiers
- [ ] Creer le `.gitignore`
- [ ] Creer le `requirements.txt`
- [ ] Dezipper et preparer le dataset

### Phase 2 -- Exploration et nettoyage
- [ ] Notebook `00_exploration_nettoyage.ipynb`
  - Chargement de `train.csv`
  - Inspection : colonnes, types, valeurs manquantes, doublons
  - Distribution des 6 labels, co-occurrence entre labels
  - Longueur des commentaires (distribution, outliers)
  - Desequilibre de classes
  - Wordclouds par categorie de toxicite
  - Nettoyage : HTML, caracteres speciaux, doublons, lignes vides
  - Sauvegarde du dataset nettoye

### Phase 3 -- Split et preprocessing
- [ ] Script `src/preprocessing.py`
  - Split de `train.csv` nettoye en train (70%) / val (15%) / test (15%)
  - Split stratifie (respecte la distribution des labels)
  - Seed fixe pour reproductibilite
  - Sauvegarde dans `data/processed/` (train.csv, val.csv, test.csv)
  - `test.csv` Kaggle non touche, reserve pour submission

### Phase 4 -- Baselines
- [ ] Notebook `01_baselines.ipynb`
  - TF-IDF + Logistic Regression
  - TF-IDF + SVM
  - TF-IDF + Naive Bayes
  - BERT (bert-base-uncased) fine-tune -- baseline deep learning
  - Entrainement sur train, tuning sur val, evaluation sur test
  - Sauvegarde des metriques et modeles

### Phase 5 -- Modeles recents
- [ ] Notebook `02_transformers.ipynb`
  - Fine-tuning DistilBERT
  - Fine-tuning ModernBERT
  - Fine-tuning NeoBERT
  - Entrainement sur train, tuning sur val, evaluation sur test
  - Sauvegarde des metriques et modeles

### Phase 6 -- Comparaison
- [ ] Notebook `03_comparaison.ipynb`
  - Tableau comparatif de toutes les metriques
  - Analyse par label
  - Matrices de confusion
  - Analyse cout/performance
  - Conclusion
  - (Optionnel) Predictions sur test.csv Kaggle au format submission

### Phase 7 -- Dashboard
- [ ] App Streamlit `dashboard/app.py`
  - Section EDA interactive
  - Section prediction de toxicite (saisie utilisateur)
  - Section comparaison des modeles
- [ ] Deploiement sur Streamlit Community Cloud

### Phase 8 -- Livrables
- [ ] Note methodologique (PDF, max 10 pages)
- [ ] Slides de soutenance (PDF, max 30 slides)
- [ ] Zip final nomme selon le format OC

---

## Structure cible du projet
```
openclassrooms-projet9/
  data/
    raw/                    # Dataset brut Jigsaw
    processed/              # Donnees nettoyees et splittees
  notebooks/
    00_exploration_nettoyage.ipynb
    01_baselines.ipynb
    02_transformers.ipynb
    03_comparaison.ipynb
  src/
    preprocessing.py
  dashboard/
    app.py
  artifacts/
    models/                 # Modeles sauvegardes
    metrics/                # Metriques JSON
    figures/                # Graphiques exportes
  reports/
    note_methodologique.md
    plan_previsionnel.md
  requirements.txt
  .gitignore
  README.md
  TODO.md
```
