# Etapes a suivre -- Projet OpenClassrooms (preuve de concept ML)

Objectif global : demontrer qu'un modele recent est plus performant qu'une baseline sur le dataset Jigsaw Toxic Comment Classification Challenge (Kaggle), puis presenter la preuve dans un notebook, une note methodologique et un dashboard.

## 1. Sujet retenu ✅
- **Dataset** : Jigsaw Toxic Comment Classification Challenge (Kaggle)
  - ~160 000 commentaires Wikipedia annotes par des humains
  - 6 labels de toxicite (multi-label) : `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
  - Fichiers : `train.csv`, `test.csv`, `test_labels.csv`, `sample_submission.csv`
- **Baselines (modeles classiques + reference deep learning)** : TF-IDF + Logistic Regression, TF-IDF + SVM, TF-IDF + Naive Bayes, BERT (bert-base-uncased)
- **Modeles recents** : DistilBERT, ModernBERT (dec 2024), NeoBERT (feb 2025)
- Critere de conformite : modeles recents (moins de 2 ans pour ModernBERT et NeoBERT).

## 2. Rediger le plan previsionnel (1 page) ✅
- [x] Decrire les modeles recents et les raisons du choix.
- [x] Decrire le dataset retenu.
- [x] Ajouter 2-3 references bibliographiques fiables.
- [x] Decrire la demarche de test (protocole de comparaison baselines vs modeles recents).
- **Fichier** : `PLAN_PREVISIONNEL.md`
- Livrable attendu : PDF du plan previsionnel.

## 3. Valider le plan avec le mentor
- [ ] Soumettre le plan.
- [ ] Noter les retours.
- [ ] Ajuster le protocole si demande avant de coder la POC.

## 4. Preparer l'environnement de travail ✅
- [x] Creer un environnement Python dedie (venv).
- [x] Installer les dependances -- 2 fichiers requirements :
  - `requirements-mac.txt` (CPU/MPS)
  - `requirements-pc.txt` (NVIDIA GPU + TensorFlow CUDA)
- [x] Organiser le dossier projet (data, notebooks, src, dashboard, artifacts).

## 5. Explorer et nettoyer le dataset ✅
- [x] Notebook `00_exploration_nettoyage.ipynb` -- execute
- **Exploration** :
  - [x] Verifier les colonnes, types, valeurs manquantes, doublons.
  - [x] Distribution des 6 labels, co-occurrence entre labels.
  - [x] Longueur des commentaires (distribution, outliers).
  - [x] Desequilibre de classes (certaines categories tres rares).
  - [x] Wordclouds par categorie de toxicite.
- **Nettoyage** :
  - [x] Suppression des doublons et lignes vides.
  - [x] Nettoyage du texte (HTML, caracteres speciaux, normalisation).
  - [x] Documentation des choix de nettoyage.
- [x] Dataset nettoye sauvegarde dans `data/processed/cleaned.csv`.
- **Figures** : `label_distribution.png`, `label_cooccurrence.png`, `text_length_distribution.png`, `wordclouds_by_label.png`

## 6. Splitter le dataset et definir le protocole d'evaluation ✅
- [x] Script `src/preprocessing.py` execute
- **Split** : `train.csv` (nettoye) en 3 parties :
  - **train** (70%) : 111 478 commentaires
  - **val** (15%) : 23 888 commentaires
  - **test** (15%) : 23 889 commentaires
  - Split stratifie, seed fixe (42).
- [x] `test.csv` Kaggle reserve, non utilise pendant l'entrainement.
- **Fichiers** : `data/processed/train.csv`, `val.csv`, `test.csv`

## 7. Implementer les baselines ✅ (execute sur machine GPU)
- [x] Notebook `01_baselines.ipynb` -- execute sur GPU (CUDA)
- [x] Pipeline 1 : TF-IDF + Logistic Regression (OneVsRest) -- ROC-AUC: 0.9786
- [x] Pipeline 2 : TF-IDF + Linear SVM (OneVsRest) -- ROC-AUC: 0.9743
- [x] Pipeline 3 : TF-IDF + Multinomial Naive Bayes (OneVsRest) -- ROC-AUC: 0.9611
- [x] Pipeline 4 : BERT (bert-base-uncased) fine-tune -- ROC-AUC: 0.9878
- **Metriques** : `artifacts/metrics/baselines_metrics.json`
- **⚠️ Modeles a recuperer** : fichiers .joblib et bert_baseline/best/ depuis la machine GPU

## 8. Implementer les modeles recents ⏳ (2/3 termine)
- Notebooks splites en 1 par modele :
  - [x] `02a_distilbert.ipynb` -- execute sur GPU -- ROC-AUC: 0.9895
  - [x] `02b_modernbert.ipynb` -- execute sur GPU -- ROC-AUC: 0.9911
  - [ ] `02c_neobert.ipynb` -- **a relancer** (metriques non sauvegardees)
- Ancien `02_transformers.ipynb` conserve comme reference
- **Metriques** : `artifacts/metrics/transformers_metrics.json` (DistilBERT + ModernBERT)
- **⚠️ Modeles a recuperer** : distilbert/best/, modernbert/best/, neobert/best/ depuis la machine GPU
- **⚠️ A FAIRE** : relancer `02c_neobert.ipynb` pour obtenir les metriques NeoBERT

## 9. Comparer baselines vs modeles recents ⏳ (pret a executer)
- [x] Notebook `03_comparaison.ipynb` -- code complet
  - Chargement flexible des metriques (fichiers combines + individuels)
  - Tableau de comparaison de toutes les metriques
  - Analyse par label (ROC-AUC par label)
  - Heatmap des metriques
  - Analyse cout/performance (scatter temps vs ROC-AUC)
  - Conclusion automatique
- **⚠️ A EXECUTER** quand les metriques NeoBERT seront disponibles

## 10. Rediger la note methodologique (max 10 pages) ❌ A FAIRE
- [ ] Expliquer la demarche, les choix techniques et la reproductibilite.
- [ ] Expliquer le fonctionnement des modeles recents (architecture Transformer, ModernBERT, NeoBERT).
- [ ] Presenter les resultats compares et les limites.
- Livrable attendu : PDF de la note methodologique.

## 11. Construire le dashboard ✅ (code termine)
- [x] App Streamlit `dashboard/app.py`
- [x] Section 1 : EDA interactive (distributions, co-occurrence, stats multi-label, longueur)
- [x] Section 2 : Prediction de toxicite (selecteur de modele, saisie utilisateur)
- [x] Section 3 : Comparaison des metriques (tableau, bar charts, radar, cout/performance, conclusion)
- [x] `dashboard/requirements.txt`
- **Strategie** :
  - Streamlit Cloud : TF-IDF + Logistic Regression (leger)
  - Local (soutenance) : Transformer via selecteur de modele

## 12. Deployer le dashboard sur le cloud ❌ A FAIRE
- [ ] Deployer sur Streamlit Community Cloud.
- [ ] Verifier que l'application est accessible et stable.
- [ ] Conserver l'URL publique pour la soutenance.
- **⚠️ Bloque par** : modele TF-IDF a recuperer depuis la machine GPU

## 13. Preparer les slides de soutenance (max 30 slides) ❌ A FAIRE
- [ ] Bloc 1 (5 min) : dataset, modeles, sources, plan.
- [ ] Bloc 2 (10 min) : demarche, concepts des modeles recents, resultats compares.
- [ ] Bloc 3 (5 min) : demonstration dashboard.
- [ ] Preparer reponses aux questions (risques, biais, limites, couts).

## 14. Preparer le depot final des livrables ❌ A FAIRE
- [ ] Generer et nommer les fichiers selon le format demande.
- [ ] Creer le zip final "Titre_du_projet_nom_prenom".
- [ ] Verifier la presence de tous les livrables : plan, notebook, note methodo, code dashboard, lien deploiement, slides PDF.

---

## Blocages actuels
1. **Modeles et metriques sur la machine GPU** : les fichiers .joblib, .safetensors et le JSON NeoBERT doivent etre recuperes
2. **NeoBERT** : le notebook a ete execute mais la cellule de sauvegarde n'a pas tourne -- a relancer
3. Tout le reste (comparaison, deploiement, livrables) depend de la recuperation des artifacts

## Criteres de reussite minimaux
- Le notebook contient bien baselines et modeles recents compares dans un meme document.
- Les modeles recents sont justifies par des sources de qualite.
- Les metriques montrent une comparaison claire et interpretable.
- Le dashboard fonctionne en cloud et illustre la preuve de concept.
