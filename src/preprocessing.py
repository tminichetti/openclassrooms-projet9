"""
Preprocessing : split du dataset nettoye en train/val/test.

Usage :
    python src/preprocessing.py

Input  : data/processed/cleaned.csv (genere par le notebook 00)
Output : data/processed/train.csv, data/processed/val.csv, data/processed/test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Config ---
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

INPUT_PATH = os.path.join('data', 'processed', 'cleaned.csv')
OUTPUT_DIR = os.path.join('data', 'processed')


def make_stratify_col(df: pd.DataFrame) -> pd.Series:
    """Cree une colonne de stratification a partir des 6 labels binaires.

    On concatene les labels en une chaine (ex: '010010') pour stratifier
    sur la combinaison de labels. Les combinaisons trop rares (< 2 occurrences)
    sont regroupees sous 'rare' pour eviter les erreurs de split.
    """
    strat = df[LABEL_COLS].astype(str).agg(''.join, axis=1)
    counts = strat.value_counts()
    rare_combos = counts[counts < 2].index
    strat = strat.replace(rare_combos, 'rare')
    return strat


def split_dataset():
    """Split le dataset nettoye en train/val/test."""
    print(f"Chargement de {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  -> {len(df)} lignes chargees.")

    strat = make_stratify_col(df)

    # Premier split : train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=strat
    )

    # Second split : val (50% du temp = 15%) vs test (50% du temp = 15%)
    strat_temp = make_stratify_col(temp_df)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=strat_temp
    )

    print(f"\n  Train : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total : {len(train_df) + len(val_df) + len(test_df)}")

    # Verification de la distribution des labels
    print("\nDistribution des labels (% positifs) :")
    print(f"  {'Label':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
    for label in LABEL_COLS:
        t = train_df[label].mean() * 100
        v = val_df[label].mean() * 100
        te = test_df[label].mean() * 100
        print(f"  {label:20s} {t:7.2f}% {v:7.2f}% {te:7.2f}%")

    # Sauvegarde
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

    print(f"\nFichiers sauvegardes dans {OUTPUT_DIR}/")
    print("  -> train.csv, val.csv, test.csv")


if __name__ == '__main__':
    split_dataset()
