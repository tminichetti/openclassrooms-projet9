# Etapes a suivre -- Projet OpenClassrooms (preuve de concept ML)

Objectif global : demontrer qu'un modele recent est plus performant qu'une baseline sur le dataset Jigsaw Toxic Comment Classification Challenge (Kaggle), puis presenter la preuve dans un notebook, une note methodologique et un dashboard.

## 1. Sujet retenu
- **Dataset** : Jigsaw Toxic Comment Classification Challenge (Kaggle)
  - ~160 000 commentaires Wikipedia annotes par des humains
  - 6 labels de toxicite (multi-label) : `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
  - Fichiers : `train.csv`, `test.csv`, `test_labels.csv`, `sample_submission.csv`
- **Baselines (modeles classiques + reference deep learning)** : TF-IDF + Logistic Regression, TF-IDF + SVM, TF-IDF + Naive Bayes, BERT (bert-base-uncased)
- **Modeles recents** : DistilBERT, ModernBERT (dec 2024), NeoBERT (feb 2025)
- Critere de conformite : modeles recents (moins de 2 ans pour ModernBERT et NeoBERT).

## 2. Rediger le plan previsionnel (1 page)
- Decrire les modeles recents et les raisons du choix.
- Decrire le dataset retenu.
- Ajouter 2-3 references bibliographiques fiables.
- Decrire la demarche de test (protocole de comparaison baselines vs modeles recents).
- Livrable attendu : PDF du plan previsionnel.

## 3. Valider le plan avec le mentor
- Soumettre le plan.
- Noter les retours.
- Ajuster le protocole si demande avant de coder la POC.

## 4. Preparer l'environnement de travail
- Creer un environnement Python dedie.
- Installer les dependances (pandas, scikit-learn, transformers, datasets, torch, evaluate, matplotlib/seaborn, streamlit).
- Organiser le dossier projet (data, notebooks, src, dashboard, reports, artifacts).

## 5. Explorer et nettoyer le dataset
- Charger `train.csv` depuis `data/raw/`.
- **Exploration** :
  - Verifier les colonnes, types, valeurs manquantes, doublons.
  - Distribution des 6 labels, co-occurrence entre labels.
  - Longueur des commentaires (distribution, outliers).
  - Desequilibre de classes (certaines categories tres rares).
  - Wordclouds par categorie de toxicite.
- **Nettoyage** :
  - Suppression des doublons et lignes vides.
  - Nettoyage du texte (HTML, caracteres speciaux, normalisation).
  - Documentation des choix de nettoyage.
- Sauvegarder le dataset nettoye dans `data/processed/`.

## 6. Splitter le dataset et definir le protocole d'evaluation
- **Split** : decouper `train.csv` (nettoye) en 3 parties :
  - **train** (70%) : entrainement des modeles
  - **val** (15%) : validation / tuning des hyperparametres
  - **test** (15%) : evaluation finale des modeles
  - Split stratifie pour respecter la distribution des labels.
  - Seed fixe pour la reproductibilite.
- **test.csv Kaggle** : reserve et non utilise pendant l'entrainement. Utilise uniquement a la fin pour simuler une submission Kaggle (coherence avec le challenge).
- **Metriques** : ROC-AUC (macro), F1-score (macro), Precision, Recall, Hamming Loss, temps d'entrainement et d'inference.
- **Reproductibilite** : seed fixe, meme preprocessing, memes splits pour tous les modeles.

## 7. Implementer les baselines
- Pipeline 1 : TF-IDF + Logistic Regression (OneVsRest)
- Pipeline 2 : TF-IDF + Linear SVM (OneVsRest)
- Pipeline 3 : TF-IDF + Multinomial Naive Bayes (OneVsRest)
- Pipeline 4 : BERT (bert-base-uncased) fine-tune -- baseline deep learning de reference
- Entrainer sur train, tuner sur val, evaluer sur test, sauvegarder les resultats.

## 8. Implementer les modeles recents
- Modele 1 : DistilBERT (distilbert-base-uncased) fine-tune
- Modele 2 : ModernBERT (answerdotai/ModernBERT-base) fine-tune -- dec 2024
- Modele 3 : NeoBERT fine-tune -- feb 2025
- Tokeniser les textes, fine-tuner sur train, tuner sur val, evaluer sur test avec les memes metriques.

## 9. Comparer baselines vs modeles recents
- Tableau de comparaison de toutes les metriques.
- Analyse par label (certains labels sont tres rares : `threat`, `identity_hate`).
- Matrice de confusion par label.
- Conclure : gain, perte, stabilite, cout de calcul.

## 10. Rediger la note methodologique (max 10 pages)
- Expliquer la demarche, les choix techniques et la reproductibilite.
- Expliquer le fonctionnement des modeles recents (architecture Transformer, ModernBERT, NeoBERT).
- Presenter les resultats compares et les limites.
- Livrable attendu : PDF de la note methodologique.

## 11. Construire le dashboard
- Creer une app Streamlit.
- Section 1 : EDA interactive (distribution des labels, wordclouds, etc.)
- Section 2 : Prediction de toxicite sur un texte saisi par l'utilisateur.
- Section 3 : Comparaison des metriques entre tous les modeles.

## 12. Deployer le dashboard sur le cloud
- Deployer sur Streamlit Community Cloud.
- Verifier que l'application est accessible et stable.
- Conserver l'URL publique pour la soutenance.

## 13. Preparer les slides de soutenance (max 30 slides)
- Bloc 1 (5 min) : dataset, modeles, sources, plan.
- Bloc 2 (10 min) : demarche, concepts des modeles recents, resultats compares.
- Bloc 3 (5 min) : demonstration dashboard.
- Preparer reponses aux questions (risques, biais, limites, couts).

## 14. Preparer le depot final des livrables
- Generer et nommer les fichiers selon le format demande.
- Creer le zip final "Titre_du_projet_nom_prenom".
- Verifier la presence de tous les livrables : plan, notebook, note methodo, code dashboard, lien deploiement, slides PDF.

## Criteres de reussite minimaux
- Le notebook contient bien baselines et modeles recents compares dans un meme document.
- Les modeles recents sont justifies par des sources de qualite.
- Les metriques montrent une comparaison claire et interpretable.
- Le dashboard fonctionne en cloud et illustre la preuve de concept.
