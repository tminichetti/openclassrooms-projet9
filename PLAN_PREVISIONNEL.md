# Plan previsionnel -- Preuve de concept

## 1) Sujet retenu
Classification multi-label de commentaires toxiques sur Wikipedia, en comparant des baselines classiques a des modeles NLP recents (Transformers).

## 2) Algorithmes envisages et justification

### Baselines (modeles classiques)
- **TF-IDF + Logistic Regression (OneVsRest)** : reference classique, rapide, interpretable.
- **TF-IDF + Linear SVM (OneVsRest)** : performant sur les taches de classification de texte a haute dimension.
- **TF-IDF + Multinomial Naive Bayes (OneVsRest)** : baseline probabiliste simple, souvent efficace en NLP.
- **BERT (bert-base-uncased)** : modele Transformer de reference (2018), sert de baseline "deep learning" pour mesurer l'apport des modeles plus recents.

### Modeles recents
- **DistilBERT (distilbert-base-uncased)** : version distillee de BERT, 40% plus legere, 60% plus rapide, avec ~97% des performances. Permet d'evaluer le compromis performance/cout.
- **ModernBERT (answerdotai/ModernBERT-base)** : encodeur bidirectionnel moderne (arXiv, 18/12/2024), entraine sur 2T tokens, sequence native 8192, SOTA sur classification et retrieval. Conforme au critere "moins de 2 ans".
- **NeoBERT (neobert-base)** : encodeur "next-gen" (arXiv, 26/02/2025), 250M params, contexte 4096, bat ModernBERT sur le benchmark MTEB. Le plus recent des trois.

### Pourquoi cette approche multi-modeles
- Comparer plusieurs baselines et plusieurs modeles recents donne une vision plus riche et credible.
- Permet d'analyser l'apport reel des Transformers vs methodes classiques, et les differences entre Transformers.

## 3) Dataset retenu pour l'evaluation
**Dataset** : Jigsaw Toxic Comment Classification Challenge (Kaggle)

**Lien** : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

**Caracteristiques** :
- ~160 000 commentaires Wikipedia annotes par des humains
- 6 labels binaires (multi-label) : `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Split train/test fourni
- Desequilibre de classes significatif (ex : `threat` et `identity_hate` tres rares)

**Raison du choix** :
- Dataset de reference en NLP pour la classification de texte.
- Tache multi-label qui presente un defi realiste (desequilibre, co-occurrence de labels).
- Grande taille permettant un entrainement fiable et une evaluation robuste.
- Bien documente et largement utilise dans la communaute.

## 4) References bibliographiques
1. **BERT** (Devlin et al., 2018) : https://arxiv.org/abs/1810.04805
2. **DistilBERT** (Sanh et al., 2019) : https://arxiv.org/abs/1910.01108
3. **ModernBERT** (Warner et al., 2024) : https://arxiv.org/abs/2412.13663
4. **NeoBERT** (Le Breton et al., 2025) : https://arxiv.org/abs/2502.19587
5. **Jigsaw Toxic Comment Challenge** (Kaggle) : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
6. **scikit-learn documentation** : https://scikit-learn.org/stable/

## 5) Demarche de test (preuve de concept)

### Objectif
Demontrer que les modeles Transformers recents ameliorent la performance de classification multi-label de toxicite par rapport aux baselines classiques TF-IDF.

### Protocole de comparaison
1. Charger et nettoyer le dataset (textes + 6 labels binaires).
2. Construire un split reproductible : train fourni -> 80% train / 20% validation ; test fourni (en excluant les lignes avec label = -1).
3. Entrainer les 4 baselines (3 TF-IDF + BERT) sur train, evaluer sur validation et test.
4. Fine-tuner les 3 modeles recents (DistilBERT, ModernBERT, NeoBERT) sur train/validation, evaluer sur test.
5. Comparer tous les modeles sur le meme jeu de test avec les memes metriques.

### Metriques prevues
- **ROC-AUC (macro)** : metrique principale (standard Kaggle pour ce challenge)
- **F1-score (macro)**
- **Precision et Recall (macro)**
- **Hamming Loss**
- **Temps d'entrainement et d'inference** (analyse cout/performance)

### Critere de succes
Les modeles Transformers doivent presenter un gain mesurable sur ROC-AUC et/ou F1-score par rapport aux baselines, avec une analyse claire du compromis performance/cout.

## 6) Reutilisation eventuelle de code externe
Si du code de tutoriel est reutilise (par exemple pipeline Hugging Face), la source sera citee explicitement dans le notebook et la note methodologique.
Le cas d'usage reste original car applique au dataset Jigsaw Toxic Comment avec une approche comparative multi-modeles.
