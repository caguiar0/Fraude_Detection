"""
=============================================================================
src/models/train.py
=============================================================================
Módulo de treinamento dos modelos de Machine Learning.

Modelos implementados:
    1. Logistic Regression  — baseline clássico, interpretável
    2. Random Forest        — ensemble de árvores, robusto
    3. XGBoost              — gradient boosting, estado da arte em tabular

Todos os modelos recebem class_weight='balanced' (quando disponível), que
faz o algoritmo penalizar mais os erros na classe minoritária (fraudes),
COMPLEMENTANDO o SMOTE aplicado anteriormente.

Estrutura retornada:
    dict com nome do modelo → objeto treinado (estimator)
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Constante de configuração
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Definição dos modelos
# ─────────────────────────────────────────────────────────────────────────────
def _build_models() -> Dict[str, Any]:
    """
    Constrói e retorna os modelos com seus hiperparâmetros configurados.

    Cada modelo inclui comentários explicando as escolhas de hiperparâmetros.

    Returns:
        dict: { 'nome_amigável': estimator_sklearn }
    """
    models = {

        # ── 1. Logistic Regression ────────────────────────────────────────────
        # • Modelo linear: separa as classes com um hiperplano
        # • class_weight='balanced': aumenta a importância das fraudes
        # • max_iter=1000: aumentado para garantir convergência em dados grandes
        # • solver='lbfgs': eficiente para datasets médios/grandes
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,  # usa todos os núcleos de CPU disponíveis
        ),

        # ── 2. Random Forest ──────────────────────────────────────────────────
        # • Ensemble de 100 árvores de decisão independentes
        # • Cada árvore vê uma amostra aleatória dos dados (bagging)
        # • class_weight='balanced_subsample': rebalanceia dentro de cada árvore
        # • n_estimators=100: bom equilíbrio entre performance e tempo
        # • max_depth=None: árvores crescem até folhas puras (pode overfitar)
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            max_features="sqrt",   # raiz quadrada das features em cada split
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # ── 3. XGBoost ────────────────────────────────────────────────────────
        # • Gradient Boosting: constrói árvores sequencialmente, cada uma
        #   corrigindo os erros da anterior
        # • scale_pos_weight: ajusta o peso da classe positiva (fraude)
        #   Valor 578 ≈ proporção original (578 legítimas para 1 fraude)
        #   Mesmo com SMOTE aplicado, manter este parâmetro ajuda o modelo
        # • eval_metric='aucpr': otimiza a área sob a curva Precision-Recall
        #   ideal para dados desbalanceados
        # • use_label_encoder=False: evita warning de deprecação
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=578,  # proporção classe majoritária / minoritária
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,           # silencia logs do XGBoost
        ),
    }

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Função principal
# ─────────────────────────────────────────────────────────────────────────────
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """
    Treina todos os modelos definidos e retorna um dicionário com os
    estimadores treinados.

    Args:
        X_train: Features de treino (pós-SMOTE)
        y_train: Labels de treino (pós-SMOTE)

    Returns:
        dict: { 'nome_do_modelo': estimator_treinado }
    """
    import time

    models = _build_models()
    trained_models = {}

    print("\n" + "=" * 60)
    print("  TREINAMENTO DOS MODELOS")
    print("=" * 60)
    print(f"  Amostras de treino: {len(X_train):,}")
    print(f"  Features:           {X_train.shape[1]}")
    print(f"  Fraudes no treino:  {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n[🤖] Treinando: {name}...")
        start_time = time.time()

        # ── Treina o modelo ───────────────────────────────────────────────────
        model.fit(X_train, y_train)

        elapsed = time.time() - start_time
        trained_models[name] = model

        print(f"[✓] {name} treinado em {elapsed:.1f}s")

    print(f"\n[✓] Todos os {len(trained_models)} modelos treinados com sucesso!\n")
    return trained_models
