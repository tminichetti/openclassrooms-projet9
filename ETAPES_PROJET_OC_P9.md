# Étapes à suivre — Projet OpenClassrooms (preuve de concept ML)

Objectif global : démontrer qu'un modèle récent est plus performant qu'une baseline sur un dataset choisi, puis présenter la preuve dans un notebook, une note méthodologique et un dashboard.

## 1. Confirmer le sujet final
- Choisir le trio final : dataset, baseline, nouveau modèle.
- Sujet recommandé ici : Inflation Research Abstracts (UCI) + baseline TF-IDF/Logistic Regression + ModernBERT.
- Critère de conformité : modèle récent (moins de 5 ans dans la mission, ici moins de 2 ans).

## 2. Rédiger le plan prévisionnel (1 page)
- Décrire le modèle récent et les raisons du choix.
- Décrire le dataset retenu.
- Ajouter 2-3 références bibliographiques fiables.
- Décrire la démarche de test (protocole de comparaison baseline vs nouveau modèle).
- Livrable attendu : PDF du plan prévisionnel.

## 3. Valider le plan avec le mentor
- Soumettre le plan.
- Noter les retours.
- Ajuster le protocole si demandé avant de coder la POC.

## 4. Préparer l'environnement de travail
- Créer un environnement Python dédié.
- Installer les dépendances utiles (pandas, scikit-learn, transformers, datasets, torch, evaluate, matplotlib/seaborn, streamlit).
- Organiser le dossier projet (data, notebooks, src, dashboard, reports, slides).

## 5. Charger et explorer le dataset
- Télécharger et versionner localement le dataset.
- Vérifier les colonnes, la qualité des textes, la répartition des classes, les valeurs manquantes/doublons.
- Faire une mini EDA orientée modélisation (longueur des textes, distribution labels).

## 6. Définir un protocole d'évaluation strict
- Fixer les métriques : Accuracy, Macro-F1, Precision, Recall (et éventuellement temps d'entraînement/inférence).
- Fixer le split (train/validation/test) de façon identique pour les deux modèles.
- Fixer la reproductibilité (seed, même prétraitement logique, mêmes données).

## 7. Implémenter la baseline
- Construire pipeline baseline : TF-IDF + Logistic Regression.
- Entraîner sur train, régler quelques hyperparamètres simples sur validation.
- Évaluer sur test et sauvegarder les résultats.

## 8. Implémenter le nouveau modèle (ModernBERT)
- Tokeniser les textes avec le tokenizer ModernBERT.
- Fine-tuner le modèle sur train/validation.
- Évaluer sur test avec les mêmes métriques.
- Sauvegarder les résultats et les paramètres d'entraînement.

## 9. Comparer baseline vs modèle récent
- Faire un tableau de comparaison des métriques.
- Ajouter une matrice de confusion et une analyse d'erreurs.
- Conclure clairement : gain, perte, stabilité, coût de calcul.

## 10. Rédiger la note méthodologique (max 10 pages)
- Expliquer la démarche, les choix techniques et la reproductibilité.
- Expliquer le fonctionnement du nouveau modèle (ModernBERT) et son apport.
- Présenter les résultats comparés et les limites.
- Livrable attendu : PDF de la note méthodologique.

## 11. Construire le dashboard
- Créer une app Streamlit.
- Montrer des prédictions sur exemples texte.
- Montrer les métriques comparées baseline vs nouveau modèle.
- Ajouter une section "Conclusion" claire pour la démonstration.

## 12. Déployer le dashboard sur le cloud
- Déployer (exemple : Streamlit Community Cloud).
- Vérifier que l'application est accessible et stable.
- Conserver l'URL publique pour la soutenance.

## 13. Préparer les slides de soutenance (max 30 slides)
- Bloc 1 (5 min) : dataset, modèle, sources, plan.
- Bloc 2 (10 min) : démarche, concepts du nouveau modèle, résultats comparés.
- Bloc 3 (5 min) : démonstration dashboard.
- Préparer réponses aux questions (risques, biais, limites, coûts).

## 14. Préparer le dépôt final des livrables
- Générer et nommer les fichiers selon le format demandé.
- Créer le zip final "Titre_du_projet_nom_prénom".
- Vérifier la présence de tous les livrables : plan, notebook, note méthodo, code dashboard, lien déploiement, slides PDF.

## Critères de réussite minimaux
- Le notebook contient bien baseline et nouveau modèle comparés dans un même document.
- Le modèle récent est justifié par des sources de qualité.
- Les métriques montrent une comparaison claire et interprétable.
- Le dashboard fonctionne en cloud et illustre la preuve de concept.
