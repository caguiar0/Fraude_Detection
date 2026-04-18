# -*- coding: utf-8 -*-
"""
app.py — Dashboard Web (Streamlit) para Detecção de Fraudes
Execute com: streamlit run app.py
"""

import sys, os, io, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f0f1a; }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a, #16213e);
        border: 1px solid #30305a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7c83fd; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
    .stTabs [data-baseweb="tab"] { color: #aaa; font-size: 1rem; }
    .stTabs [aria-selected="true"] { color: #7c83fd !important; border-bottom: 2px solid #7c83fd; }
    h1, h2, h3 { color: #e0e0ff !important; }
    p, li { color: #ccc; }
</style>
""", unsafe_allow_html=True)

# ── Imports dos módulos do projeto ────────────────────────────────────────────
from src.data.download import download_dataset
from src.data.preprocessing import preprocess
from src.features.engineering import apply_smote
from src.models.train import train_all_models
from src.models.evaluate import evaluate_all_models
from sklearn.metrics import roc_curve, auc

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Fraud Detection")
    st.markdown("---")
    st.markdown("### ⚙️ Configurações")

    smote_strategy = st.slider(
        "SMOTE – proporção fraudes/legítimas",
        min_value=0.1, max_value=1.0, value=0.5, step=0.1,
        help="Ex: 0.5 = fraudes serão 50% do total de legítimas após SMOTE"
    )
    test_size = st.slider(
        "Tamanho do conjunto de teste (%)",
        min_value=10, max_value=40, value=20, step=5
    )
    n_estimators = st.slider("Random Forest – nº de árvores", 50, 300, 100, 50)

    st.markdown("---")
    run_btn = st.button("🚀 Executar Pipeline", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Dataset:** Credit Card Fraud\n\n**Fonte:** OpenML #1597\n\n**Registros:** 284.807\n\n**Fraudes:** 492 (0,17%)")

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔍 Detecção de Fraudes em Transações Bancárias")
st.markdown("Dashboard interativo com análise exploratória, treinamento e avaliação de modelos de ML.")
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE — mantém resultados entre reruns
# ═════════════════════════════════════════════════════════════════════════════
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
if run_btn:
    progress = st.progress(0, text="Carregando dataset...")

    # 1. Download
    df = download_dataset()
    progress.progress(15, text="Dataset carregado!")

    # 2. Pré-processamento
    from src.data.preprocessing import TARGET_COLUMN, POSSIBLE_SCALE_COLS, RANDOM_STATE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df_clean = df.drop_duplicates()
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=RANDOM_STATE, stratify=y
    )
    cols = [c for c in POSSIBLE_SCALE_COLS if c in X_train.columns]
    if cols:
        sc = StandardScaler()
        X_train[cols] = sc.fit_transform(X_train[cols])
        X_test[cols]  = sc.transform(X_test[cols])
    progress.progress(30, text="Pré-processamento concluído!")

    # 3. SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE, k_neighbors=5)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_train, y_train)
    X_tr_bal = pd.DataFrame(X_tr_bal, columns=X_train.columns)
    y_tr_bal = pd.Series(y_tr_bal, name=y_train.name)
    progress.progress(50, text="SMOTE aplicado!")

    # 4. Treinamento
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, scale_pos_weight=578, eval_metric="aucpr", use_label_encoder=False, random_state=RANDOM_STATE, verbosity=0),
    }
    trained = {}
    for i, (name, m) in enumerate(models.items()):
        progress.progress(55 + i * 10, text=f"Treinando {name}...")
        m.fit(X_tr_bal, y_tr_bal)
        trained[name] = m
    progress.progress(85, text="Avaliando modelos...")

    # 5. Avaliação
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, average_precision_score, confusion_matrix)
    all_res, metrics_rows = {}, []
    for name, m in trained.items():
        yp = m.predict(X_test)
        yproba = m.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, yp)
        tn, fp, fn, tp = cm.ravel()
        all_res[name] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                         "_y_test": y_test, "_y_proba": yproba}
        metrics_rows.append({
            "Modelo": name,
            "Precision": round(precision_score(y_test, yp), 4),
            "Recall":    round(recall_score(y_test, yp), 4),
            "F1-Score":  round(f1_score(y_test, yp), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, yproba), 4),
            "PR-AUC":    round(average_precision_score(y_test, yproba), 4),
        })

    metrics_df = pd.DataFrame(metrics_rows).set_index("Modelo")
    progress.progress(100, text="Pipeline concluído!")

    # Salva no session_state
    st.session_state.results_ready = True
    st.session_state.df         = df_clean
    st.session_state.metrics_df = metrics_df
    st.session_state.all_res    = all_res
    st.session_state.y_test     = y_test

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Análise Exploratória (EDA)", "🤖 Resultados dos Modelos", "📖 Sobre o Projeto"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # Carrega o dataset só para EDA (mesmo sem rodar o pipeline)
    @st.cache_data(show_spinner="Carregando dataset para EDA...")
    def load_df():
        return download_dataset().drop_duplicates()

    df_eda = load_df()

    # KPIs
    fraud  = df_eda[df_eda["Class"] == 1]
    legit  = df_eda[df_eda["Class"] == 0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_eda):,}</div><div class="metric-label">Total de Transações</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(fraud):,}</div><div class="metric-label">Transações Fraudulentas</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{fraud["Amount"].mean():.2f}€</div><div class="metric-label">Valor Médio de Fraude</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">0.17%</div><div class="metric-label">Taxa de Fraude</div></div>', unsafe_allow_html=True)

    st.markdown("### Distribuição das Classes")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(
            x=["Legítima (0)", "Fraude (1)"],
            y=[len(legit), len(fraud)],
            color=["Legítima (0)", "Fraude (1)"],
            color_discrete_map={"Legítima (0)": "#2ecc71", "Fraude (1)": "#e74c3c"},
            text_auto=True, template="plotly_dark",
            labels={"x": "Classe", "y": "Contagem"},
            title="Contagem Absoluta"
        )
        fig_bar.update_traces(textfont_size=14)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_b:
        fig_pie = px.pie(
            values=[len(legit), len(fraud)],
            names=["Legítima (0)", "Fraude (1)"],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            template="plotly_dark", title="Proporção Percentual",
            hole=0.45
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Distribuição do Valor (Amount) por Classe")
    fig_hist = px.histogram(
        df_eda, x="Amount", color=df_eda["Class"].map({0: "Legítima", 1: "Fraude"}),
        nbins=80, barmode="overlay", opacity=0.7, template="plotly_dark",
        color_discrete_map={"Legítima": "#2ecc71", "Fraude": "#e74c3c"},
        log_y=True, labels={"color": "Classe"},
        title="Distribuição do Valor da Transação (escala log)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Heatmap de Correlação (amostra de 5.000 linhas)")
    sample = df_eda.sample(5000, random_state=42)
    corr   = sample.drop(columns=["Class"]).corr().round(2)
    fig_heat = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template="plotly_dark", title="Correlação de Pearson entre Features",
        aspect="auto"
    )
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Resultados
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.results_ready:
        st.info("👈 Clique em **Executar Pipeline** na barra lateral para treinar os modelos.")
    else:
        mdf     = st.session_state.metrics_df
        all_res = st.session_state.all_res

        # Comparativo de métricas
        st.markdown("### 📊 Comparativo de Métricas")
        cols_m = st.columns(len(mdf))
        colors_map = {"Logistic Regression": "#3498db", "Random Forest": "#e67e22", "XGBoost": "#9b59b6"}
        for col, (model, row) in zip(cols_m, mdf.iterrows()):
            with col:
                c = colors_map.get(model, "#7c83fd")
                st.markdown(f"""
                <div class="metric-card" style="border-color:{c}55">
                    <div class="metric-value" style="color:{c}">{row['F1-Score']:.3f}</div>
                    <div class="metric-label">F1-Score</div>
                    <hr style="border-color:{c}33;margin:8px 0">
                    <div style="color:#aaa;font-size:.8rem">
                        Precision: {row['Precision']:.3f}<br>
                        Recall: {row['Recall']:.3f}<br>
                        ROC-AUC: {row['ROC-AUC']:.3f}<br>
                        PR-AUC: {row['PR-AUC']:.3f}
                    </div>
                    <div style="margin-top:8px;font-weight:700;color:{c}">{model}</div>
                </div>""", unsafe_allow_html=True)

        # Tabela
        st.markdown("### 📋 Tabela de Métricas")
        st.dataframe(mdf.style.highlight_max(axis=0, color="#1e3a2e")
                     .format("{:.4f}"), use_container_width=True)

        # Barplot
        st.markdown("### 📈 Gráfico Comparativo")
        metrics_long = mdf[["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]].reset_index().melt(
            id_vars="Modelo", var_name="Métrica", value_name="Score"
        )
        fig_bar2 = px.bar(
            metrics_long, x="Métrica", y="Score", color="Modelo",
            barmode="group", template="plotly_dark",
            color_discrete_map=colors_map,
            title="Comparativo de Todas as Métricas por Modelo"
        )
        fig_bar2.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig_bar2, use_container_width=True)

        # Curvas ROC
        st.markdown("### 📉 Curvas ROC")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        for name, res in all_res.items():
            fpr, tpr, _ = roc_curve(res["_y_test"], res["_y_proba"])
            roc_auc_val = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc_val:.4f})",
                line=dict(width=2.5, color=colors_map.get(name))
            ))
        fig_roc.update_layout(
            template="plotly_dark",
            title="Curvas ROC — Comparativo de Modelos",
            xaxis_title="Taxa de Falsos Positivos (FPR)",
            yaxis_title="Taxa de Verdadeiros Positivos (TPR)",
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion matrices
        st.markdown("### 🔢 Confusion Matrices")
        cm_cols = st.columns(len(all_res))
        for col, (name, res) in zip(cm_cols, all_res.items()):
            with col:
                cm_data = [[res["TN"], res["FP"]], [res["FN"], res["TP"]]]
                fig_cm = px.imshow(
                    cm_data,
                    labels=dict(x="Predito", y="Real", color="Count"),
                    x=["Legítima", "Fraude"], y=["Legítima", "Fraude"],
                    color_continuous_scale="Blues", template="plotly_dark",
                    title=name, text_auto=True
                )
                fig_cm.update_layout(height=300)
                st.plotly_chart(fig_cm, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Sobre
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## 📖 Sobre o Projeto

    Este projeto é um pipeline educacional de **Machine Learning para detecção de fraudes bancárias**.

    ### Dataset
    - **Credit Card Fraud Detection** — ULB Machine Learning Group
    - 284.807 transações europeias (setembro de 2013)
    - Features V1–V28: resultado de PCA (anonimizadas)
    - Fonte: [OpenML #1597](https://www.openml.org/d/1597)

    ### Modelos
    | Modelo | Tipo | Vantagem |
    |---|---|---|
    | Logistic Regression | Linear | Rápido, interpretável, baseline |
    | Random Forest | Ensemble (Bagging) | Robusto, lida bem com ruído |
    | XGBoost | Ensemble (Boosting) | Estado da arte para tabular |

    ### Por que usar SMOTE?
    O dataset é **extremamente desbalanceado** (~578 legítimas para 1 fraude).
    O SMOTE gera amostras **sintéticas** da classe minoritária interpolando entre vizinhos reais,
    permitindo que os modelos aprendam padrões de fraude sem simplesmente "ignorar" a classe rara.

    ### Por que não usar Accuracy?
    Um modelo que chuta sempre **"legítima"** acerta **99.83%** — mas detecta **zero fraude**.
    As métricas corretas são: **Recall** (fraudes detectadas), **Precision** (alarmes falsos) e **F1-Score**.

    ### Estrutura do projeto
    ```
    fraud_detection/
    ├── src/data/         → download e pré-processamento
    ├── src/features/     → SMOTE e feature engineering
    ├── src/models/       → treinamento e avaliação
    ├── src/visualization/→ gráficos matplotlib (para main.py)
    ├── app.py            → este dashboard Streamlit
    └── main.py           → pipeline via terminal
    ```
    """)
