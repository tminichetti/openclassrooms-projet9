# Plan previsionnel -- Preuve de concept

## 1) Sujet retenu
Classification multi-label de commentaires toxiques sur Wikipedia, en comparant des baselines classiques a des modeles NLP recents (Transformers).

## 2) Algorithmes envisages et justification

### Baseline classique
- **TF-IDF + Logistic Regression (OneVsRest)** : reference classique, rapide, interpretable.

### Baseline deep learning
- **BERT (bert-base-uncased)** : modele Transformer de reference (2018), sert de baseline "deep learning" pour mesurer l'apport des modeles plus recents.

### Modele recent
- **ModernBERT (answerdotai/ModernBERT-base)** : encodeur bidirectionnel moderne (arXiv, 18/12/2024), entraine sur 2T tokens, sequence native 8192, SOTA sur classification et retrieval. Conforme au critere "moins de 5 ans".

### Pourquoi ces 3 modeles
- TF-IDF + LogReg : baseline classique permettant de mesurer l'apport des Transformers.
- BERT : baseline deep learning de reference, point de comparaison standard.
- ModernBERT : architecture recente (dec 2024) permettant de demontrer le gain des avancees recentes.

## 3) Dataset retenu pour l'evaluation
**Dataset** : Jigsaw Toxic Comment Classification Challenge (Kaggle)

**Lien** : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

**Caracteristiques** :
- ~160 000 commentaires Wikipedia annotes par des humains
- 6 labels binaires (multi-label) : `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- `train.csv` : donnees labelisees (sera splitte en train/val/test par nos soins)
- `test.csv` : donnees sans labels, reserve pour simulation de submission Kaggle
- Desequilibre de classes significatif (ex : `threat` et `identity_hate` tres rares)

**Raison du choix** :
- Dataset de reference en NLP pour la classification de texte.
- Tache multi-label qui presente un defi realiste (desequilibre, co-occurrence de labels).
- Grande taille permettant un entrainement fiable et une evaluation robuste.
- Bien documente et largement utilise dans la communaute.

## 4) References bibliographiques
1. **BERT** (Devlin et al., 2018) : https://arxiv.org/abs/1810.04805
2. **ModernBERT** (Warner et al., 2024) : https://arxiv.org/abs/2412.13663
3. **Jigsaw Toxic Comment Challenge** (Kaggle) : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
4. **scikit-learn documentation** : https://scikit-learn.org/stable/

## 5) Demarche de test (preuve de concept)

### Objectif
Demontrer que les modeles Transformers recents ameliorent la performance de classification multi-label de toxicite par rapport aux baselines classiques TF-IDF.

### Protocole de comparaison
1. **Explorer et nettoyer** le dataset : analyse qualite, doublons, texte vide, nettoyage HTML/caracteres speciaux.
2. **Splitter `train.csv`** (nettoye) en 3 parties reproductibles (seed fixe, stratifie) :
   - train (70%) : entrainement
   - val (15%) : validation / hyperparametres
   - test (15%) : evaluation finale
3. **`test.csv` Kaggle** : reserve, non utilise pendant l'entrainement. Utilise uniquement a la fin pour simuler une submission coherente avec le challenge.
4. Entrainer les 2 baselines (TF-IDF LogReg + BERT) sur train, tuner sur val, evaluer sur test.
5. Fine-tuner ModernBERT sur train, tuner sur val, evaluer sur test.
6. Comparer tous les modeles sur le meme jeu de test avec les memes metriques.
7. (Optionnel) Generer les predictions sur `test.csv` Kaggle au format submission.

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
