# Idées de projet (datasets + méthodes récentes)

## 1. NLP — Classification d’abstracts avec ModernBERT
Dataset (nom + date + lien) : Inflation Research Abstracts (UCI), publié le 16/02/2025, lien : [UCI Inflation Research Abstracts](https://archive.ics.uci.edu/dataset/1125/inflation+research+abstracts+classification)
Nouveau modèle (nom + date + lien) : ModernBERT, publié le 18/12/2024, lien : [ModernBERT (arXiv)](https://arxiv.org/abs/2412.13663)
Baseline proposée : TF-IDF + Logistic Regression.
Métriques de comparaison : Accuracy, Macro-F1, Precision, Recall, temps d’entraînement et temps d’inférence.
Pourquoi c’est un bon sujet OC : POC rapide à exécuter, gains souvent visibles face à une baseline classique, démonstration claire en soutenance.

## 2. Tabulaire santé — Toxicité médicamenteuse avec TabM
Dataset (nom + date + lien) : Drug Induced Autoimmunity Prediction (UCI), publié le 05/01/2025, lien : [UCI Drug Induced Autoimmunity](https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction)
Nouveau modèle (nom + date + lien) : TabM, publié le 31/10/2024, lien : [TabM (arXiv)](https://arxiv.org/abs/2410.24210)
Baseline proposée : XGBoost ou RandomForest.
Métriques de comparaison : ROC-AUC, PR-AUC, F1-score, Recall, calibration (Brier score).
Pourquoi c’est un bon sujet OC : Cas d’usage santé pertinent, comparaison tabulaire moderne vs méthodes standards très défendable.

## 3. Cybersécurité — Détection de phishing URL avec TabM
Dataset (nom + date + lien) : PhiUSIIL Phishing URL Dataset (UCI), publié le 03/03/2024, lien : [UCI PhiUSIIL Phishing URL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
Nouveau modèle (nom + date + lien) : TabM, publié le 31/10/2024, lien : [TabM (arXiv)](https://arxiv.org/abs/2410.24210)
Baseline proposée : XGBoost.
Métriques de comparaison : F1-score, Recall, Precision, ROC-AUC, faux positifs par classe.
Pourquoi c’est un bon sujet OC : Sujet concret et compréhensible, métriques sécurité faciles à présenter, dataset exploitable rapidement.

## 4. Time series énergie — Forecasting avec Chronos-2
Dataset (nom + date + lien) : PGCB Hourly Generation Dataset (Bangladesh) (UCI), publié le 17/06/2025, lien : [UCI PGCB Hourly Generation](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh))
Nouveau modèle (nom + date + lien) : Chronos-2, publié le 17/10/2025, lien : [Chronos-2 (arXiv)](https://arxiv.org/abs/2510.15821). Alternative récente : TabPFN-TS, publié le 06/01/2025, lien : [TabPFN-TS (arXiv)](https://arxiv.org/abs/2501.02945)
Baseline proposée : Prophet.
Métriques de comparaison : MAE, RMSE, MAPE, quantile loss (si prévision probabiliste), temps d’inférence.
Pourquoi c’est un bon sujet OC : Excellent effet “state of the art” pour le dashboard, mais mise en oeuvre un peu plus technique.

## Meilleur choix recommandé
Choix: 1. NLP — ModernBERT + Inflation Research Abstracts
Justification: rapide à entraîner, POC clair, bonnes chances de gain vs TF-IDF+LR, soutenance plus facile.

## Références (liens officiels)
[UCI Inflation Research Abstracts](https://archive.ics.uci.edu/dataset/1125/inflation+research+abstracts+classification)
[UCI Drug Induced Autoimmunity](https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction)
[UCI PhiUSIIL Phishing URL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
[UCI PGCB Hourly Generation](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh))
[ModernBERT (arXiv)](https://arxiv.org/abs/2412.13663)
[TabM (arXiv)](https://arxiv.org/abs/2410.24210)
[Chronos-2 (arXiv)](https://arxiv.org/abs/2510.15821)
[TabPFN-TS (arXiv)](https://arxiv.org/abs/2501.02945)
[ModernBERT GitHub](https://github.com/AnswerDotAI/ModernBERT)
[TabM GitHub](https://github.com/yandex-research/tabm)
[Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
[TabPFN-TS GitHub](https://github.com/PriorLabs/tabpfn-time-series)
