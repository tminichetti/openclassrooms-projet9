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

### Baseline classique
| Modele | Description |
|--------|-------------|
| TF-IDF + Logistic Regression | Reference classique, interpretable |

### Baseline deep learning
| Modele | Description |
|--------|-------------|
| BERT (bert-base-uncased) | Baseline deep learning de reference (2018) |

### Modele recent (Transformer)
| Modele | Description | Ref |
|--------|-------------|-----|
| ModernBERT (answerdotai/ModernBERT-base) | Encodeur moderne, 2T tokens, seq 8192 | arXiv Dec 2024 |

## Metriques
- ROC-AUC (macro) -- metrique principale
- F1-score (macro)
- Precision / Recall (macro)
- Hamming Loss
- Temps d'entrainement / inference

---

## Taches

### Phase 1 -- Setup ✅ TERMINEE
- [x] Creer la branche de travail depuis main
- [x] Mettre en place la structure de dossiers
- [x] Creer le `.gitignore`
- [x] Creer le `requirements.txt` (requirements-mac.txt + requirements-pc.txt)
- [x] Dezipper et preparer le dataset

### Phase 2 -- Exploration et nettoyage ✅ TERMINEE
- [x] Notebook `00_exploration_nettoyage.ipynb`
  - Chargement de `train.csv`
  - Inspection : colonnes, types, valeurs manquantes, doublons
  - Distribution des 6 labels, co-occurrence entre labels
  - Longueur des commentaires (distribution, outliers)
  - Desequilibre de classes
  - Wordclouds par categorie de toxicite
  - Nettoyage : HTML, caracteres speciaux, doublons, lignes vides
  - Sauvegarde du dataset nettoye
- **Figures generees** : `label_distribution.png`, `label_cooccurrence.png`, `text_length_distribution.png`, `wordclouds_by_label.png`

### Phase 3 -- Split et preprocessing ✅ TERMINEE
- [x] Script `src/preprocessing.py`
  - Split de `train.csv` nettoye en train (70%) / val (15%) / test (15%)
  - Split stratifie (respecte la distribution des labels)
  - Seed fixe pour reproductibilite
  - Sauvegarde dans `data/processed/` (train.csv, val.csv, test.csv, cleaned.csv)
  - `test.csv` Kaggle non touche, reserve pour submission
- **Fichiers generes** : `data/processed/train.csv` (111 478), `val.csv` (23 888), `test.csv` (23 889)

### Phase 4 -- Baselines ✅ TERMINEE (execution GPU)
- [x] Notebook `01_baselines.ipynb` -- execute sur machine GPU (CUDA)
  - [x] TF-IDF + Logistic Regression -- ROC-AUC: 0.9786, F1: 0.4667
  - [x] BERT (bert-base-uncased) fine-tune -- ROC-AUC: 0.9878, F1: 0.6965
  - Entrainement sur train, tuning sur val, evaluation sur test
  - Sauvegarde des metriques et modeles
- **⚠️ Artifacts a recuperer depuis la machine GPU** :
  - `artifacts/models/baseline_tfidf_logreg.joblib`
  - `artifacts/models/tfidf_vectorizer.joblib`
  - `artifacts/models/bert_baseline/best/`
- **Metriques JSON recrees** : `artifacts/metrics/baselines_metrics.json` (extraites des outputs du notebook)

### Phase 5 -- Modele recent ✅ TERMINEE (execution GPU)
- [x] `02b_modernbert.ipynb` -- execute sur GPU -- ROC-AUC: 0.9911, F1: 0.6632
- **⚠️ Artifacts a recuperer depuis la machine GPU** :
  - `artifacts/models/modernbert/best/`
- **Metriques JSON** : `artifacts/metrics/transformers_metrics.json`

### Phase 6 -- Comparaison ⏳ PRET A EXECUTER
- [x] Notebook `03_comparaison.ipynb` -- code complet, pret a executer
  - Tableau comparatif des 3 modeles
  - Analyse par label (ROC-AUC par label)
  - Heatmap des metriques
  - Analyse cout/performance (scatter temps vs ROC-AUC)
  - Conclusion automatique (meilleur modele, gain vs baseline)
  - Sauvegarde de `all_metrics.json` et `comparaison_globale.csv`

### Phase 7 -- Dashboard ✅ CODE TERMINE
- [x] App Streamlit `dashboard/app.py` -- 3 sections :
  - Section EDA interactive (distributions, co-occurrence, stats multi-label, longueur)
  - Section comparaison des modeles (tableau, bar charts, radar, cout/performance, conclusion)
  - Section prediction de toxicite (selecteur de modele, saisie utilisateur)
- [x] `dashboard/requirements.txt`
- **Strategie de deploiement** :
  - Streamlit Cloud : TF-IDF + Logistic Regression (leger, ~50 Mo)
  - Local (soutenance) : Transformer (BERT ou ModernBERT) pour la demo
- [ ] Deploiement sur Streamlit Community Cloud
- **⚠️ Bloque par** : modeles a recuperer depuis la machine GPU (logreg + BERT + ModernBERT)

### Phase 8 -- Livrables ❌ A FAIRE
- [x] Plan previsionnel (`PLAN_PREVISIONNEL.md`) -- redige
- [ ] Note methodologique (PDF, max 10 pages)
- [ ] Slides de soutenance (PDF, max 30 slides)
- [ ] Zip final nomme selon le format OC

---

---

## Resultats intermediaires (metriques sur test set)

| Modele | Type | ROC-AUC (macro) | F1 (macro) | Precision | Recall | Hamming Loss | Train (s) | Inference (s) |
|--------|------|-----------------|------------|-----------|--------|--------------|-----------|---------------|
| TF-IDF + Logistic Regression | Baseline | 0.9786 | 0.4667 | 0.7831 | 0.3536 | 0.0199 | 15.88 | 0.04 |
| BERT (bert-base-uncased) | Baseline | 0.9878 | 0.6965 | 0.6767 | 0.7240 | 0.0153 | 2933 | 59.27 |
| ModernBERT | Recent | **0.9911** | **0.6632** | 0.7081 | 0.6350 | 0.0153 | 4836 | 120.18 |

---

## Structure actuelle du projet
```
openclassrooms-projet9/
  data/
    raw/                    # Dataset brut Jigsaw (.zip + CSV)
    processed/              # train.csv, val.csv, test.csv, cleaned.csv
  notebooks/
    00_exploration_nettoyage.ipynb   # ✅ Execute
    01_baselines.ipynb               # ✅ Execute (GPU) -- TF-IDF LogReg + BERT
    02b_modernbert.ipynb             # ✅ Execute (GPU) -- ModernBERT
    03_comparaison.ipynb             # Pret a executer
  src/
    preprocessing.py
  dashboard/
    app.py                  # ✅ App Streamlit (EDA, comparaison, prediction)
    requirements.txt
  artifacts/
    models/                 # ⚠️ A recuperer depuis machine GPU
    metrics/                # baselines_metrics.json, transformers_metrics.json
    figures/                # 4 figures EDA
  MISSION.md
  LIVRABLES.md
  ETAPES_PROJET_OC_P9.md
  PLAN_PREVISIONNEL.md
  TODO.md
  requirements-mac.txt
  requirements-pc.txt
  .gitignore
```
