import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -- Config --
st.set_page_config(
    page_title="Toxic Comment Classification",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
FIGURES_DIR = BASE_DIR / "artifacts" / "figures"

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# -- Helper functions --
@st.cache_data
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, val, test


@st.cache_data
def load_metrics():
    all_results = []
    for fname in ["baselines_metrics.json", "transformers_metrics.json"]:
        path = METRICS_DIR / fname
        if path.exists():
            with open(path) as f:
                all_results.extend(json.load(f))
    return all_results


# -- Sidebar --
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    ["EDA", "Comparaison des modeles", "Prediction"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Projet** : Classification multi-label de commentaires toxiques\n\n"
    "**Dataset** : Jigsaw Toxic Comment (Kaggle)\n\n"
    "**Modeles** : TF-IDF LogReg, BERT, ModernBERT"
)


# ========================================================
# PAGE 1 : EDA
# ========================================================
if page == "EDA":
    st.title("Analyse exploratoire des donnees")

    train, val, test = load_data()

    st.markdown(f"**Train** : {len(train):,} | **Val** : {len(val):,} | **Test** : {len(test):,}")

    # -- Distribution des labels --
    st.subheader("Distribution des labels")

    label_counts = train[LABEL_COLS].sum().sort_values(ascending=False)
    label_pct = (label_counts / len(train) * 100).round(2)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=label_counts.index,
            y=label_counts.values,
            labels={"x": "Label", "y": "Nombre de commentaires"},
            title="Nombre de commentaires par label",
            color=label_counts.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=label_pct.index,
            y=label_pct.values,
            labels={"x": "Label", "y": "% du dataset"},
            title="Pourcentage de commentaires par label",
            color=label_pct.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # -- Co-occurrence --
    st.subheader("Matrice de co-occurrence des labels")

    cooc = train[LABEL_COLS].T.dot(train[LABEL_COLS])
    fig = px.imshow(
        cooc,
        labels=dict(color="Co-occurrences"),
        x=LABEL_COLS,
        y=LABEL_COLS,
        color_continuous_scale="YlOrRd",
        text_auto=True,
    )
    fig.update_layout(width=700, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # -- Multi-label stats --
    st.subheader("Statistiques multi-label")

    train["num_labels"] = train[LABEL_COLS].sum(axis=1)
    label_dist = train["num_labels"].value_counts().sort_index()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=label_dist.index.astype(str),
            y=label_dist.values,
            labels={"x": "Nombre de labels", "y": "Nombre de commentaires"},
            title="Distribution du nombre de labels par commentaire",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        clean_pct = (train["num_labels"] == 0).mean() * 100
        toxic_pct = (train["num_labels"] > 0).mean() * 100
        multi_pct = (train["num_labels"] > 1).mean() * 100

        st.metric("Commentaires non-toxiques", f"{clean_pct:.1f}%")
        st.metric("Commentaires toxiques", f"{toxic_pct:.1f}%")
        st.metric("Commentaires multi-label", f"{multi_pct:.1f}%")

    # -- Longueur des commentaires --
    st.subheader("Distribution de la longueur des commentaires")

    train["text_length"] = train["comment_text"].str.len()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            train,
            x="text_length",
            nbins=100,
            title="Longueur des commentaires (caracteres)",
            labels={"text_length": "Longueur"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stats = train["text_length"].describe().round(1)
        st.dataframe(stats.to_frame("Statistiques"), use_container_width=True)


# ========================================================
# PAGE 2 : Comparaison des modeles
# ========================================================
elif page == "Comparaison des modeles":
    st.title("Comparaison des modeles")

    all_results = load_metrics()

    if not all_results:
        st.warning("Aucun fichier de metriques trouve. Executez les notebooks d'entrainement d'abord.")
        st.stop()

    df = pd.DataFrame(all_results)

    # Type de modele
    baseline_names = [
        "TF-IDF + Logistic Regression",
        "BERT (bert-base-uncased)",
    ]
    df["type"] = df["model"].apply(lambda x: "Baseline" if x in baseline_names else "Transformer recent")

    # -- Tableau global --
    st.subheader("Tableau comparatif")

    display_cols = [
        "model", "type", "roc_auc_macro", "f1_macro",
        "precision_macro", "recall_macro", "hamming_loss",
        "train_time_sec", "inference_time_sec",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols].sort_values("roc_auc_macro", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # -- ROC-AUC bar chart --
    st.subheader("ROC-AUC (macro) par modele")

    df_sorted = df.sort_values("roc_auc_macro", ascending=True)
    fig = px.bar(
        df_sorted,
        x="roc_auc_macro",
        y="model",
        color="type",
        orientation="h",
        color_discrete_map={"Baseline": "#3498db", "Transformer recent": "#e74c3c"},
        text=df_sorted["roc_auc_macro"].round(4),
        labels={"roc_auc_macro": "ROC-AUC (macro)", "model": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # -- F1-score bar chart --
    st.subheader("F1-score (macro) par modele")

    df_sorted_f1 = df.sort_values("f1_macro", ascending=True)
    fig = px.bar(
        df_sorted_f1,
        x="f1_macro",
        y="model",
        color="type",
        orientation="h",
        color_discrete_map={"Baseline": "#3498db", "Transformer recent": "#e74c3c"},
        text=df_sorted_f1["f1_macro"].round(4),
        labels={"f1_macro": "F1-score (macro)", "model": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # -- Radar chart --
    st.subheader("Profil des modeles (radar)")

    radar_metrics = ["roc_auc_macro", "f1_macro", "precision_macro", "recall_macro"]
    radar_labels = ["ROC-AUC", "F1", "Precision", "Recall"]

    available_radar = [m for m in radar_metrics if m in df.columns]
    if len(available_radar) == len(radar_metrics):
        fig = go.Figure()
        for _, row in df.iterrows():
            values = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_labels + [radar_labels[0]],
                name=row["model"],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.3, 1.0])),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # -- Cout vs Performance --
    st.subheader("Cout vs Performance")

    if "train_time_sec" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df,
                x="train_time_sec",
                y="roc_auc_macro",
                color="type",
                text="model",
                size_max=15,
                color_discrete_map={"Baseline": "#3498db", "Transformer recent": "#e74c3c"},
                labels={
                    "train_time_sec": "Temps d'entrainement (s)",
                    "roc_auc_macro": "ROC-AUC (macro)",
                },
                title="Temps d'entrainement vs ROC-AUC",
            )
            fig.update_traces(textposition="top center", marker=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                df,
                x="inference_time_sec",
                y="roc_auc_macro",
                color="type",
                text="model",
                size_max=15,
                color_discrete_map={"Baseline": "#3498db", "Transformer recent": "#e74c3c"},
                labels={
                    "inference_time_sec": "Temps d'inference (s)",
                    "roc_auc_macro": "ROC-AUC (macro)",
                },
                title="Temps d'inference vs ROC-AUC",
            )
            fig.update_traces(textposition="top center", marker=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)

    # -- Conclusion --
    st.subheader("Conclusion")

    best_overall = df.loc[df["roc_auc_macro"].idxmax()]
    best_baseline = df[df["type"] == "Baseline"].loc[
        df[df["type"] == "Baseline"]["roc_auc_macro"].idxmax()
    ]
    best_recent = df[df["type"] == "Transformer recent"]
    if not best_recent.empty:
        best_recent = best_recent.loc[best_recent["roc_auc_macro"].idxmax()]
        gain = best_recent["roc_auc_macro"] - best_baseline["roc_auc_macro"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Meilleur modele", best_overall["model"], f"ROC-AUC: {best_overall['roc_auc_macro']:.4f}")
        col2.metric("Meilleure baseline", best_baseline["model"], f"ROC-AUC: {best_baseline['roc_auc_macro']:.4f}")
        col3.metric("Gain Transformer vs Baseline", f"+{gain:.4f}", "ROC-AUC")


# ========================================================
# PAGE 3 : Prediction
# ========================================================
elif page == "Prediction":
    st.title("Prediction de toxicite")
    st.markdown("Entrez un commentaire pour obtenir une prediction de toxicite.")

    # Check if a transformer model is available
    model_dirs = [
        ("BERT", BASE_DIR / "artifacts" / "models" / "bert_baseline" / "best"),
        ("ModernBERT", BASE_DIR / "artifacts" / "models" / "modernbert" / "best"),
    ]

    available_models = []
    for name, path in model_dirs:
        if path.exists() and any(path.iterdir()):
            available_models.append((name, path))

    # Also check TF-IDF fallback
    tfidf_path = BASE_DIR / "artifacts" / "models" / "tfidf_vectorizer.joblib"
    lr_path = BASE_DIR / "artifacts" / "models" / "baseline_tfidf_logreg.joblib"
    tfidf_available = tfidf_path.exists() and lr_path.exists()

    if not available_models and not tfidf_available:
        st.warning(
            "Aucun modele sauvegarde trouve dans `artifacts/models/`. "
            "Executez les notebooks d'entrainement pour sauvegarder les modeles, "
            "puis relancez le dashboard."
        )
        st.stop()

    # Model selection
    model_options = [name for name, _ in available_models]
    if tfidf_available:
        model_options.append("TF-IDF + Logistic Regression")

    selected = st.selectbox("Modele a utiliser :", model_options)

    user_input = st.text_area(
        "Votre commentaire :",
        height=150,
        placeholder="Type your comment here...",
    )

    if st.button("Analyser", type="primary"):
        if not user_input.strip():
            st.warning("Veuillez entrer un commentaire.")
            st.stop()

        if selected == "TF-IDF + Logistic Regression":
            import joblib

            @st.cache_resource
            def load_tfidf_model():
                tfidf = joblib.load(tfidf_path)
                model = joblib.load(lr_path)
                return tfidf, model

            tfidf, model = load_tfidf_model()
            X = tfidf.transform([user_input])
            probs = model.predict_proba(X)[0]

        else:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_path = dict(available_models)[selected]

            @st.cache_resource
            def load_transformer_model(path_str):
                tokenizer = AutoTokenizer.from_pretrained(path_str)
                model = AutoModelForSequenceClassification.from_pretrained(path_str)
                model.eval()
                return tokenizer, model

            tokenizer, model = load_transformer_model(str(model_path))

            encoding = tokenizer(
                user_input,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = model(**encoding).logits
                probs = torch.sigmoid(logits).squeeze().numpy()

        # Display results
        st.subheader("Resultats")
        cols = st.columns(3)
        for i, label in enumerate(LABEL_COLS):
            col = cols[i % 3]
            prob = float(probs[i])
            col.metric(label, f"{prob:.1%}")

        max_idx = int(np.argmax(probs))
        if probs[max_idx] > 0.5:
            st.error(f"Ce commentaire est probablement **{LABEL_COLS[max_idx]}** ({probs[max_idx]:.1%})")
        else:
            st.success("Ce commentaire ne semble pas toxique.")
