# Guide de présentation — Soutenance Projet 9

> Lien Canva : https://www.canva.com/d/ov7eA6SdBEarAZ4
> Durée totale estimée : **20 minutes** (5 + 10 + 5)

---

## Slide 1 — Titre

**Contenu actuel :** Classification de commentaires toxiques / Thomas Minichetti

**Ce que tu dis :**
> "Bonjour, je vais vous présenter mon projet 9 qui porte sur la classification automatique de commentaires toxiques. L'objectif est de comparer trois approches NLP : une méthode classique TF-IDF, un modèle BERT de 2018, et ModernBERT sorti en décembre 2024."

**Images :** aucune

---

## Slide 2 — Sommaire

**Contenu actuel :** 3 blocs — Dataset/modèles, Méthodologie/résultats, Dashboard

**Ce que tu dis :**
> "La présentation suit trois parties : d'abord le dataset et les modèles choisis, ensuite la méthodologie et les résultats, et enfin une démo du dashboard interactif."

**Images :** aucune

---

## Slide 3 — Dataset Jigsaw Toxic Comment

**Contenu actuel :** ~160 000 commentaires annotés par des humains

**Ce que tu dis :**
> "Le dataset vient du challenge Kaggle Jigsaw. Il contient 159 571 commentaires Wikipedia annotés par des humains pour 6 catégories de toxicité : toxic, severe_toxic, obscene, threat, insult, identity_hate. C'est un problème multi-label : un commentaire peut appartenir à plusieurs catégories simultanément. Environ 10% des commentaires sont toxiques — le dataset est donc très déséquilibré."

**Images :** aucune (texte suffit ici)

---

## Slide 4 — Exploration des données (EDA)

**Contenu actuel :** 3 placeholders textuels — Distribution, Co-occurrence, Comment Length

**Ce que tu dis :**
> "Avant de modéliser, on a exploré le dataset. Trois observations importantes."

**👉 AJOUTER DES IMAGES ICI — priorité maximale :**

| Image à glisser-déposer | Emplacement | Ce que tu montres |
|-------------------------|-------------|-------------------|
| `artifacts/figures/label_distribution.png` | Zone "Distribution" | Distribution très déséquilibrée : toxic ~10%, severe_toxic <1% |
| `artifacts/figures/label_cooccurrence.png` | Zone "Co-occurrence" | Les labels co-occurrent souvent (obscene + insult très liés) |
| `artifacts/figures/text_length_distribution.png` | Zone "Comment Length" | Les textes longs (>256 tokens) sont tronqués par BERT/ModernBERT |

**Bonus si tu veux :** `artifacts/figures/wordclouds_by_label.png` à ajouter sur un slide séparé ou en popup.

**Ce que tu dis pour chaque :**
- Distribution : *"La classe majoritaire est 'toxic' mais elle ne représente que 9.5% des données. Les classes rares comme 'threat' sont sous 0.3%."*
- Co-occurrence : *"Les labels ne sont pas indépendants. Si un commentaire est 'obscene', il est souvent aussi 'insult'."*
- Longueur : *"La plupart des commentaires font moins de 256 tokens, ce qui justifie notre choix de max_length=256 pour les transformers."*

---

## Slide 5 — Modèles sélectionnés

**Contenu actuel :** TF-IDF, BERT, ModernBERT — approches distinctes

**Ce que tu dis :**
> "Trois modèles ont été retenus pour représenter différentes générations de NLP. TF-IDF + Régression Logistique comme baseline classique rapide. BERT comme référence deep learning de 2018, état de l'art pendant plusieurs années. Et ModernBERT, architecture modernisée sortie fin 2024, qui intègre des mécanismes comme RoPE, Flash Attention 2, et GeGLU."

**Images :** aucune obligatoire (schéma optionnel si tu veux en ajouter un manuellement)

---

## Slide 6 — Sources et références

**Contenu actuel :** Devlin 2018 (BERT), Warner 2024 (ModernBERT), Vaswani 2017 (Transformers), Kaggle Jigsaw

**Ce que tu dis :**
> "Rapidement les références — les papiers originaux BERT et ModernBERT, l'article Attention is All You Need qui fonde l'architecture Transformer, et le challenge Kaggle qui fournit le dataset."

**Images :** aucune
**Durée :** 30 secondes max, slide de transition

---

## Slide 7 — Protocole expérimental

**Contenu actuel :** Split 70/15/15, ROC-AUC macro, métriques complémentaires F1/précision/rappel

**Ce que tu dis :**
> "Le protocole est identique pour les trois modèles pour garantir une comparaison équitable. Split stratifié 70/15/15 avec seed fixé à 42 — 'stratifié' signifie que la proportion de chaque label est préservée dans chaque split. La métrique principale est le ROC-AUC macro, qui est la métrique officielle du challenge Kaggle et qui gère bien le déséquilibre de classes. On mesure aussi F1, précision et rappel comme métriques complémentaires."

**Images :** aucune obligatoire

**👉 Optionnel :** ajouter un schéma simple du split (70/15/15 en barres) si tu veux rendre ça visuel — ça prend 2 minutes à faire dans Canva directement.

---

## Slide 8 — TF-IDF + Logistic Regression

**Contenu actuel :** Description de la méthode vectorisation → classification

**Ce que tu dis :**
> "Le modèle baseline : on vectorise chaque commentaire avec TF-IDF (fréquence des mots pondérée par leur rareté dans le corpus), puis on entraîne une régression logistique en mode OneVsRest — une régression par label. C'est simple, interprétable, et s'entraîne en 19 secondes. Ce modèle établit le plancher de performance à battre."

**Images :** aucune

---

## Slide 9 — BERT

**Contenu actuel :** Architecture bidirectionnelle, self-attention, révolution NLP 2018

**Ce que tu dis :**
> "BERT de Google (2018) : premier modèle à utiliser l'attention bidirectionnelle — il lit le contexte gauche ET droite simultanément. Fine-tuné sur nos données pendant 3 époques avec une tête de classification multi-label. Entraînement : ~49 minutes sur GPU."

**Images :** aucune obligatoire
**Optionnel :** schema de l'architecture BERT (disponible en ligne, ou créer un schéma simple dans Canva)

---

## Slide 10 — ModernBERT

**Contenu actuel :** Pré-entraîné sur 2 trillions de tokens, performances inégalées

**Ce que tu dis :**
> "ModernBERT d'Answer.ai (décembre 2024) : même concept que BERT mais architecture modernisée. Il intègre RoPE (positional encoding rotatif qui gère mieux les longues séquences), Flash Attention 2 (attention optimisée GPU), et GeGLU (fonction d'activation plus performante). Pré-entraîné sur 2 trillions de tokens vs 3.3 milliards pour BERT. Entraînement sur nos données : ~84 minutes."

**Images :** aucune

---

## Slide 11 — Tableau des résultats

**Contenu actuel :** Tableau modèles / métriques (placeholders)

**👉 AJOUTER LES VRAIES VALEURS dans le tableau Canva :**

| Modèle | ROC-AUC | F1-macro | Précision | Rappel | Train (s) |
|--------|---------|----------|-----------|--------|-----------|
| TF-IDF + LogReg | **0.9786** | 0.4666 | 0.7831 | 0.3536 | 19s |
| BERT | 0.9878 | **0.6965** | 0.6767 | 0.7240 | 2933s |
| ModernBERT | **0.9914** | 0.6427 | 0.7410 | 0.5749 | 5057s |

**Ce que tu dis :**
> "Voici les résultats sur le jeu de test. ModernBERT domine en ROC-AUC avec 0.9914. Mais chose surprenante, BERT obtient le meilleur F1-score à 0.697, devant ModernBERT à 0.643."

---

## Slide 12 — Comparaison des performances (graphiques)

**Contenu actuel :** "Comparaison ROC-AUC / F1-Score Analyse" — placeholders

**👉 AJOUTER DES IMAGES ICI depuis le notebook 03 :**

Les graphiques sont générés dans `notebooks/03_comparaison.ipynb`. Pour les obtenir :
1. Ouvrir le notebook
2. Exécuter et prendre des screenshots des cellules graphiques :
   - **Bar chart ROC-AUC** (les 3 modèles côte à côte)
   - **Bar chart F1-score** (les 3 modèles côte à côte)

**Ce que tu dis :**
> "Visuellement on voit bien la hiérarchie. Sur ROC-AUC, la progression est nette : TF-IDF 0.979 → BERT 0.988 → ModernBERT 0.991. Sur F1, la surprise : BERT 0.697 dépasse ModernBERT 0.643."

---

## Slide 13 — Coût et performance des modèles

**Contenu actuel :** 3 insights — TF-IDF rapidité, ModernBERT performance, équilibre

**Ce que tu dis :**
> "Ce slide résume le trade-off. TF-IDF s'entraîne en 19 secondes — imbattable pour les ressources. BERT offre le meilleur compromis F1/coût. ModernBERT donne la meilleure ROC-AUC mais au prix de 84 minutes d'entraînement. Le choix du modèle dépend du contexte : production temps-réel → TF-IDF ou BERT, détection la plus précise → ModernBERT."

**👉 Optionnel :** scatter plot temps vs ROC-AUC avec les 3 points — très lisible et impactant, tu peux le créer rapidement dans le notebook ou dans Excel.

---

## Slide 14 — Interprétation globale

**Contenu actuel :** "Les Transformers apportent un gain significatif en classification toxique"

**Ce que tu dis :**
> "Pourquoi F1 et ROC-AUC divergent ? Le F1 dépend du seuil de décision fixé à 0.5. BERT est plus 'direct' dans ses prédictions, avec un meilleur rappel (0.724 vs 0.575 pour ModernBERT). ModernBERT discrimine mieux (ROC-AUC) mais avec un seuil à 0.5 il est plus conservateur. Si on optimisait le seuil par label, ModernBERT aurait probablement aussi le meilleur F1. Pour répondre à la mission : les modèles récents (Transformers) surpassent clairement l'approche TF-IDF en ROC-AUC (+1.3 pts), confirmant la valeur des nouvelles architectures pour des tâches NLP complexes."

**Images :** aucune

---

## Slide 15 — Limites et améliorations

**Contenu actuel :** Seuil 0.5 non optimal, hyperparamètres non optimisés, max_length=256

**Ce que tu dis :**
> "Trois limites principales. Un : le seuil à 0.5 n'est pas optimal — on pourrait optimiser le seuil par label avec une recherche sur le jeu de validation. Deux : les hyperparamètres (learning rate, batch size, époques) n'ont pas été cherchés avec un grid search. Trois : max_length=256 tronque certains commentaires longs — ModernBERT peut aller jusqu'à 8192 tokens, ce qu'on n'exploite pas."

**Images :** aucune

---

## Slide 16 — Démonstration Dashboard

**Contenu actuel :** Interactive EDA, Model Comparison, Real-Time Predictions

**Ce que tu dis :**
> "Je vais maintenant faire une démo live du dashboard Streamlit. Il permet de visualiser les données EDA, comparer les modèles, et tester des prédictions en temps réel."

**👉 ACTION : Lancer le dashboard avant la soutenance**
```bash
streamlit run dashboard/app.py
```

**Images :** screenshot du dashboard à ajouter sur ce slide si tu veux (optionnel)

**Ce que tu montres dans la démo :**
1. Onglet EDA → montrer les graphiques de distribution
2. Onglet Comparaison → tableau et graphiques des 3 modèles
3. Onglet Prédiction → taper un commentaire toxique et voir le résultat

---

## Slide 17 — Citation/Conclusion

**Contenu actuel :** "ModernBERT outperforms baselines in ROC-AUC" / "The rise of Transformers revolutionized NLP models"

**Ce que tu dis :**
> "En conclusion : les Transformers ont révolutionné le NLP. ModernBERT représente l'état de l'art actuel avec des gains mesurables, mais BERT reste une référence efficace. La classification toxique est un cas d'usage réel et utile pour la modération de contenu en ligne."

**Images :** aucune

---

## Slide 18 — Merci et Questions

**Contenu actuel :** Contact, email, réseaux

**Ce que tu dis :**
> "Merci pour votre attention. Je suis disponible pour répondre à vos questions."

**👉 Mettre à jour les coordonnées** (remplacer `hello@reallygreatsite.com` et `@reallygreatsite` par tes vraies infos si tu veux)

---

## Slide 19 — (vide / de réserve)

Peut être supprimé ou utilisé comme slide bonus pour les questions.

---

## Récapitulatif des images à ajouter manuellement

| Slide | Fichier | Source |
|-------|---------|--------|
| 4 | `artifacts/figures/label_distribution.png` | notebook 01 |
| 4 | `artifacts/figures/label_cooccurrence.png` | notebook 01 |
| 4 | `artifacts/figures/text_length_distribution.png` | notebook 01 |
| 12 | Screenshot bar chart ROC-AUC | notebook 03, à capturer |
| 12 | Screenshot bar chart F1-score | notebook 03, à capturer |
| 11 | Valeurs du tableau à saisir manuellement | cf. tableau ci-dessus |

---

## Questions fréquentes à préparer

**Q : Pourquoi ModernBERT a un F1 plus bas que BERT ?**
R : Seuil de décision fixé à 0.5. ModernBERT discrimine mieux (ROC-AUC) mais ses probabilités brutes sont plus proches du seuil. BERT a un meilleur rappel (0.724 vs 0.575) ce qui fait monter le F1 avec un seuil fixe.

**Q : Pourquoi seulement 3 époques pour les transformers ?**
R : Empiriquement suffisant pour la convergence sur ce dataset (test sur val loss). Plus d'époques → risque de surapprentissage sur les classes majoritaires.

**Q : Le dataset est déséquilibré, comment vous gérez ça ?**
R : Split stratifié qui préserve les proportions. ROC-AUC insensible au déséquilibre. Pour aller plus loin : class weights, oversampling SMOTE.

**Q : ModernBERT vaut vraiment le coût de 84 minutes ?**
R : Dépend du contexte. En prod avec volume élevé → oui, le gain en précision vaut l'investissement d'entraînement (one-time cost). L'inférence reste rapide.

**Q : Pourquoi pas d'autres modèles ?**
R : Choix délibéré pour V1 : couvrir 3 générations (classique / BERT 2018 / BERT 2024) avec des résultats propres. La V2 pourrait inclure DistilBERT ou RoBERTa.
