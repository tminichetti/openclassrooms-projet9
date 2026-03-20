# Idees de projet explorees

Ce fichier archive les pistes evaluees avant de fixer le choix final.

## Pistes evaluees

| # | Domaine | Dataset | Modele recent | Baseline | Statut |
|---|---------|---------|---------------|----------|--------|
| 1 | NLP - Toxicite | Jigsaw Toxic Comment (Kaggle) | ModernBERT | TF-IDF + LogReg + BERT | **RETENU** |
| 2 | NLP - Abstracts | Inflation Research Abstracts (UCI) | ModernBERT | TF-IDF + LogReg | Ecarte |
| 3 | Tabulaire sante | Drug Induced Autoimmunity (UCI) | TabM | XGBoost / RandomForest | Ecarte |
| 4 | Cybersecurite | PhiUSIIL Phishing URL (UCI) | TabM | XGBoost | Ecarte |
| 5 | Time series | PGCB Hourly Generation (UCI) | Chronos-2 | Prophet | Ecarte |

## Justification du choix final
- **Jigsaw Toxic Comment** : dataset de reference en NLP, grande taille (~160k commentaires), tache multi-label realiste avec desequilibre de classes.
- Permet une comparaison claire : baseline classique (TF-IDF) vs baseline deep learning (BERT) vs modele recent (ModernBERT).
- Bien documente, largement utilise dans la communaute, resultats facilement comparables.

## References utiles
- [Jigsaw Toxic Comment Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [BERT (arXiv)](https://arxiv.org/abs/1810.04805)
- [ModernBERT (arXiv)](https://arxiv.org/abs/2412.13663)
- [TabM (arXiv)](https://arxiv.org/abs/2410.24210)
- [Chronos-2 (arXiv)](https://arxiv.org/abs/2510.15821)
