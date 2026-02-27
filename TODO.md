# TODO -- Plan de travail du projet

## Projet
**Classification multi-label de commentaires toxiques** (Jigsaw Toxic Comment Classification Challenge, Kaggle).
Objectif : demontrer qu'un modele NLP recent surpasse une baseline classique.

## Dataset
- **Source** : Kaggle - Jigsaw Toxic Comment Classification Challenge
- **Taille** : ~160 000 commentaires Wikipedia
- **Labels** (6, binaires, multi-label) : `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Fichiers** : `train.csv`, `test.csv`, `test_labels.csv`, `sample_submission.csv`
- **Localisation** : `data/raw/jigsaw-toxic-comment-classification-challenge/`

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

### Phase 2 -- EDA
- [ ] Notebook `00_eda.ipynb` : exploration du dataset
  - Distribution des labels
  - Co-occurrence des labels
  - Longueur des commentaires
  - Desequilibre de classes
  - Wordclouds par categorie

### Phase 3 -- Preprocessing
- [ ] Script `src/preprocessing.py`
  - Nettoyage de texte
  - Split train/val/test
  - Sauvegarde des datasets preprocesses

### Phase 4 -- Baselines
- [ ] Notebook `01_baselines.ipynb`
  - TF-IDF + Logistic Regression
  - TF-IDF + SVM
  - TF-IDF + Naive Bayes
  - BERT (bert-base-uncased) fine-tune -- baseline deep learning
  - Evaluation sur test set
  - Sauvegarde des metriques et modeles

### Phase 5 -- Modeles recents
- [ ] Notebook `02_transformers.ipynb`
  - Fine-tuning DistilBERT
  - Fine-tuning ModernBERT
  - Fine-tuning NeoBERT
  - Evaluation sur test set
  - Sauvegarde des metriques et modeles

### Phase 6 -- Comparaison
- [ ] Notebook `03_comparaison.ipynb`
  - Tableau comparatif de toutes les metriques
  - Analyse par label
  - Matrices de confusion
  - Analyse cout/performance
  - Conclusion

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
    00_eda.ipynb
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
