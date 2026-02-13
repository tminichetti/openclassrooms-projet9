# Plan prévisionnel — Preuve de concept

## 1) Sujet retenu
Amélioration d'un modèle de classification de textes académiques sur l'inflation en comparant une baseline classique à un modèle NLP récent.

## 2) Algorithme envisagé et justification
Nouveau modèle envisagé : ModernBERT (publication arXiv du 18/12/2024).

Arguments justifiant ce choix :
- Modèle récent, conforme au cadre de la mission (moins de 5 ans ; ici moins de 2 ans).
- Architecture pensée pour de bonnes performances NLP tout en restant exploitable en fine-tuning.
- Pertinent pour une tâche de classification de textes, où une baseline bag-of-words atteint souvent ses limites.

Baseline de comparaison :
- TF-IDF + Logistic Regression.

Pourquoi cette baseline :
- Référence classique, simple, rapide à entraîner, très utilisée pour les tâches de classification de texte.
- Permet une comparaison claire et pédagogique avec un modèle transformer moderne.

## 3) Dataset retenu pour l'évaluation
Dataset : Inflation Research Abstracts Classification (UCI), publié le 16/02/2025.

Lien officiel :
- https://archive.ics.uci.edu/dataset/1125/inflation+research+abstracts+classification

Raison du choix :
- Jeu de données textuel aligné avec l'objectif de classification NLP.
- Dataset récent et public, donc pertinent pour une preuve de concept académique.

## 4) Références bibliographiques
1. ModernBERT (arXiv, 2024) : https://arxiv.org/abs/2412.13663
2. UCI Inflation Research Abstracts Dataset (2025) : https://archive.ics.uci.edu/dataset/1125/inflation+research+abstracts+classification
3. Baseline TF-IDF / Logistic Regression (scikit-learn docs) : https://scikit-learn.org/stable/

## 5) Démarche de test (preuve de concept)
Objectif de la POC :
Démontrer que ModernBERT améliore la performance de classification par rapport à la baseline TF-IDF + Logistic Regression.

Protocole de comparaison :
1. Charger et nettoyer le dataset (textes + labels).
2. Construire un split reproductible train/validation/test identique pour les deux approches.
3. Entraîner la baseline TF-IDF + Logistic Regression sur train ; ajuster les paramètres sur validation.
4. Entraîner ModernBERT en fine-tuning sur train ; ajuster sur validation.
5. Évaluer les deux modèles sur le même jeu de test final.
6. Comparer les résultats avec les mêmes métriques.

Métriques prévues :
- Accuracy
- Macro-F1 (métrique principale)
- Precision
- Recall
- Temps d'entraînement et temps d'inférence (analyse coût/performance)

Critère de succès :
Le modèle ModernBERT doit présenter un gain mesurable sur Macro-F1 et/ou Recall par rapport à la baseline, avec une analyse claire du compromis performance/coût.

## 6) Réutilisation éventuelle de code externe
Si du code de tutoriel est réutilisé (par exemple pipeline Hugging Face), la source sera citée explicitement dans le notebook et la note méthodologique.
Le cas d'usage restera original car appliqué à ce dataset UCI spécifique.
