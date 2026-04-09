# Note methodologique -- Preuve de concept

## Classification multi-label de commentaires toxiques : comparaison de baselines et d'un modele Transformer recent (ModernBERT)

---

## 1. Contexte et objectif

Cette note presente la preuve de concept realisee dans le cadre d'une veille technologique en data science. L'objectif est de demontrer qu'un modele NLP recent surpasse des baselines classiques sur une tache de classification de texte.

**Tache retenue** : classification multi-label de commentaires toxiques sur Wikipedia. Chaque commentaire peut etre associe a un ou plusieurs des 6 labels de toxicite : `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

**Dataset** : Jigsaw Toxic Comment Classification Challenge (Kaggle), compose d'environ 160 000 commentaires annotes par des humains.

**Modeles compares** :
- **TF-IDF + Logistic Regression** : baseline classique (approche "sac de mots")
- **BERT (bert-base-uncased)** : baseline deep learning, Transformer de reference (2018)
- **ModernBERT (answerdotai/ModernBERT-base)** : encodeur Transformer recent (decembre 2024)

---

## 2. Demarche mise en oeuvre

### 2.1 Exploration et nettoyage des donnees

Le dataset brut (`train.csv`, 159 571 commentaires) a ete explore puis nettoye :
- Suppression des doublons et lignes vides
- Nettoyage du texte : balises HTML, caracteres speciaux, normalisation

**Observations cles** :
- Fort desequilibre de classes : ~90% des commentaires sont non-toxiques
- Les labels `threat` (0.3%) et `identity_hate` (0.9%) sont tres rares
- Forte co-occurrence entre `toxic`, `obscene` et `insult`

### 2.2 Strategie de split

Le dataset nettoye a ete divise en 3 parties avec un split stratifie (seed fixe a 42 pour la reproductibilite) :

| Partition | Taille | Usage |
|-----------|--------|-------|
| Train | 111 478 (70%) | Entrainement des modeles |
| Validation | 23 888 (15%) | Tuning des hyperparametres |
| Test | 23 889 (15%) | Evaluation finale |

Le fichier `test.csv` de Kaggle a ete reserve et non utilise pendant l'entrainement.

### 2.3 Metriques d'evaluation

- **ROC-AUC (macro)** : metrique principale, standard pour ce challenge Kaggle. Mesure la capacite du modele a distinguer les classes, independamment du seuil de decision.
- **F1-score (macro)** : equilibre entre precision et recall, sensible au desequilibre.
- **Precision et Recall (macro)** : pour analyser le compromis entre faux positifs et faux negatifs.
- **Hamming Loss** : proportion de labels mal predits.
- **Temps d'entrainement et d'inference** : pour l'analyse cout/performance.

### 2.4 Protocole experimental

Tous les modeles ont ete entraines sur le meme jeu d'entrainement, tunes sur le meme jeu de validation et evalues sur le meme jeu de test, avec les memes metriques. L'entrainement a ete realise sur GPU (NVIDIA, CUDA).

---

## 3. Modeles et algorithmes

### 3.1 Baseline classique : TF-IDF + Logistic Regression

**TF-IDF (Term Frequency - Inverse Document Frequency)** transforme le texte en vecteurs numeriques en pondérant chaque terme par sa frequence dans le document et son caractere discriminant dans le corpus. Les parametres utilises :
- `max_features=50000`, `ngram_range=(1,2)`, `sublinear_tf=True`

**Logistic Regression (OneVsRest)** : un classifieur lineaire est entraine independamment pour chaque label (approche One-vs-Rest). C'est une methode simple, rapide et interpretable, souvent utilisee comme reference en classification de texte.

**Limites** : TF-IDF ne capture pas le contexte ni l'ordre des mots. "This is not toxic" et "This is toxic" auront des representations similaires.

### 3.2 Baseline deep learning : BERT (2018)

**BERT (Bidirectional Encoder Representations from Transformers)** est un modele de langage pre-entraine introduit par Devlin et al. (2018). Il est fonde sur l'architecture Transformer (Vaswani et al., 2017).

**Concepts cles** :
- **Self-attention** : mecanisme permettant a chaque token de "regarder" tous les autres tokens de la sequence pour construire une representation contextuelle. Contrairement a TF-IDF, le mot "toxic" aura une representation differente selon son contexte.
- **Pre-entrainement bidirectionnel** : BERT est pre-entraine sur un large corpus (BookCorpus + Wikipedia, 3.3 milliards de mots) avec deux taches : Masked Language Modeling (predire des mots masques) et Next Sentence Prediction.
- **Fine-tuning** : le modele pre-entraine est ensuite adapte a la tache specifique (ici, classification multi-label) en ajoutant une couche de classification et en re-entrainant l'ensemble sur nos donnees.

**Configuration** :
- Modele : `bert-base-uncased` (110M parametres, 12 couches, 768 dimensions)
- 3 epochs, batch size 32, learning rate 2e-5, max length 256 tokens
- Optimiseur : AdamW avec weight decay 0.01
- Mixed precision (FP16) pour accelerer l'entrainement

### 3.3 Modele recent : ModernBERT (decembre 2024)

**ModernBERT** (Warner et al., 2024) est un encodeur bidirectionnel de nouvelle generation, publie en decembre 2024 sur arXiv (2412.13663). Il modernise l'architecture BERT en integrant les avancees des 6 dernieres annees.

**Nouveautes et specificites** :
- **Pre-entrainement massif** : entraine sur 2 trillions de tokens (vs 3.3 milliards pour BERT), couvrant des donnees web, du code source et des documents scientifiques.
- **Sequence native de 8192 tokens** : contre 512 pour BERT, permettant de traiter des documents beaucoup plus longs.
- **Rotary Positional Embeddings (RoPE)** : remplace les embeddings positionnels absolus de BERT. RoPE encode la position relative des tokens dans l'attention, ce qui generalise mieux aux sequences longues.
- **Flash Attention** : implementation optimisee du mecanisme d'attention, reduisant la complexite memoire et accelerant le calcul.
- **Alternance d'attention globale et locale** : les couches alternent entre une attention sur toute la sequence et une attention limitee a une fenetre locale, reduisant le cout computationnel.
- **GeGLU activation** : remplace le GELU classique de BERT par une activation avec gating, ameliorant l'expressivite du modele.
- **Pas de padding dans les batchs** : optimisation de l'entrainement en supprimant les tokens de padding inutiles.

**Pourquoi ModernBERT pour cette preuve de concept** :
- Publie en decembre 2024, conforme au critere "moins de 5 ans"
- Reference sur arXiv (2412.13663) et sites specialises
- SOTA sur plusieurs benchmarks de classification et retrieval (MTEB)
- Meme paradigme que BERT (encodeur bidirectionnel + fine-tuning), ce qui permet une comparaison directe et equitable

**Configuration** : identique a BERT (3 epochs, batch size 16, learning rate 2e-5, max length 256) pour une comparaison equitable.

---

## 4. Resultats

### 4.1 Tableau comparatif

| Modele | Type | ROC-AUC | F1 | Precision | Recall | Hamming Loss | Train (s) | Inference (s) |
|--------|------|---------|----|-----------|--------|--------------|-----------|---------------|
| TF-IDF + LogReg | Baseline classique | 0.9786 | 0.4666 | 0.7831 | 0.3536 | 0.0199 | 19 | 0.05 |
| BERT | Baseline deep learning | 0.9878 | 0.6965 | 0.6767 | 0.7240 | 0.0153 | 2933 | 59 |
| **ModernBERT** | **Recent** | **0.9914** | 0.6427 | 0.7410 | 0.5749 | 0.0157 | 5057 | 131 |

### 4.2 Analyse des resultats

**Sur la metrique principale (ROC-AUC macro)** :
- ModernBERT obtient le meilleur score (**0.9914**), devant BERT (0.9878) et TF-IDF (0.9786).
- Le gain de ModernBERT vs BERT est de **+0.36 points**, et vs TF-IDF de **+1.28 points**.
- Les scores ROC-AUC eleves (>0.97) sont coherents avec le leaderboard Kaggle de ce challenge et s'expliquent par le fort desequilibre du dataset (~90% non-toxique).

**Sur le F1-score (macro)** :
- BERT obtient le meilleur F1 (**0.6965**), devant ModernBERT (0.6427) et TF-IDF (0.4666).
- BERT a un recall nettement superieur (0.724 vs 0.575) : il detecte davantage de commentaires toxiques.
- ModernBERT a une meilleure precision (0.741 vs 0.677) : il est plus conservateur, avec moins de faux positifs mais plus de faux negatifs.

**Interpretation** : la divergence ROC-AUC / F1 s'explique par le seuil de decision fixe a 0.5. ModernBERT produit de meilleurs scores de probabilite (ranking), mais avec un seuil par defaut, il classe moins de commentaires comme toxiques. Un tuning du seuil de decision sur le jeu de validation pourrait ameliorer son F1.

### 4.3 Analyse cout / performance

| Modele | Train (s) | Inference (s) | ROC-AUC |
|--------|-----------|---------------|---------|
| TF-IDF + LogReg | 19 | 0.05 | 0.9786 |
| BERT | 2 933 | 59 | 0.9878 |
| ModernBERT | 5 057 | 131 | 0.9914 |

- TF-IDF est **~150x plus rapide** que BERT et **~260x plus rapide** que ModernBERT a l'entrainement, pour un ROC-AUC deja eleve (0.9786).
- ModernBERT est **1.7x plus lent** que BERT a l'entrainement pour un gain de +0.36 points de ROC-AUC.
- En inference, ModernBERT est **2.2x plus lent** que BERT.

### 4.4 Apport des Transformers vs methodes classiques

Le saut de performance le plus significatif est entre TF-IDF et les Transformers :
- **+0.92 points de ROC-AUC** (TF-IDF → BERT)
- **+0.23 points de F1** (TF-IDF → BERT)

Ce gain s'explique par la capacite des Transformers a capturer le contexte et les relations semantiques entre les mots, la ou TF-IDF traite chaque n-gram independamment.

Entre BERT (2018) et ModernBERT (2024), le gain est plus incremental (+0.36 ROC-AUC), ce qui confirme que les architectures Transformer ont atteint un certain plateau sur les taches de classification "classiques". Les gains des modeles recents sont plus prononces sur des taches complexes (sequences longues, retrieval), pas pleinement exploitees ici avec des commentaires courts (max 256 tokens).

---

## 5. Limites et pistes d'amelioration

### Limites de cette etude
- **Seuil de decision fixe** : le seuil de 0.5 n'est pas optimal. Un tuning par label sur le jeu de validation pourrait ameliorer significativement le F1.
- **Hyperparametres non optimises** : les 3 modeles Transformer utilisent les memes hyperparametres (3 epochs, lr 2e-5). Une recherche d'hyperparametres pourrait ameliorer les resultats.
- **Longueur de sequence** : `max_length=256` ne tire pas parti de la fenetre de 8192 tokens de ModernBERT. Sur un dataset avec des textes plus longs, le gain serait potentiellement plus marque.
- **Dataset desequilibre** : les labels rares (`threat`, `identity_hate`) restent difficiles a predire pour tous les modeles.

### Pistes d'amelioration
- Tuning du seuil de decision par label
- Augmentation de donnees pour les labels rares
- Recherche d'hyperparametres (nombre d'epochs, learning rate, batch size)
- Comparaison avec d'autres modeles recents (NeoBERT, DeBERTa-v3)
- Evaluation sur des textes longs pour exploiter la fenetre de 8192 tokens de ModernBERT

---

## 6. Conclusion

Cette preuve de concept demontre que **ModernBERT (decembre 2024) surpasse les baselines sur la metrique principale (ROC-AUC)**, confirmant l'interet des avancees recentes en NLP.

Cependant, les resultats revelent une realite nuancee :
1. **La rupture majeure en NLP reste l'arrivee des Transformers** (2018) : le gain TF-IDF → BERT est bien plus significatif que BERT → ModernBERT.
2. **Les gains entre generations de Transformers sont incrementaux** sur les taches de classification standard.
3. **Le choix du modele en production depend du cas d'usage** : TF-IDF pour la rapidite, BERT pour la detection maximale (meilleur recall), ModernBERT pour le meilleur ranking (meilleur ROC-AUC).

L'objectif du test technique est atteint : nous avons identifie, mis en oeuvre et evalue un modele recent (ModernBERT), en demontrant sa superiorite sur la metrique principale tout en analysant les compromis.

---

## 7. References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805
2. Warner, B., et al. (2024). *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*. arXiv:2412.13663
3. Vaswani, A., et al. (2017). *Attention Is All You Need*. arXiv:1706.03762
4. Jigsaw Toxic Comment Classification Challenge, Kaggle. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
5. scikit-learn documentation. https://scikit-learn.org/stable/
6. Hugging Face Transformers. https://huggingface.co/docs/transformers/
