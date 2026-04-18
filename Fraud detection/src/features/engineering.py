"""
=============================================================================
src/features/engineering.py
=============================================================================
Módulo de Feature Engineering e tratamento do desbalanceamento de classes.

Contexto do problema:
    O dataset tem ~0,17% de fraudes — para cada 578 transações legítimas,
    há apenas 1 fraude. Treinar um modelo direto nesse dado resulta em um
    classificador que "chuta" sempre 0 (legítimo) e ainda acerta 99,83%!
    Por isso, accuracy simples é INÚTIL aqui.

Estratégia: SMOTE (Synthetic Minority Over-sampling Technique)
    - Cria amostras SINTÉTICAS da classe minoritária (fraudes)
    - As amostras são geradas interpolando entre vizinhos reais
    - Aplicado APENAS no conjunto de treino (nunca no teste!)
    - O teste deve refletir a distribuição real do mundo

Alternativas ao SMOTE (não implementadas, mas válidas):
    - RandomUnderSampler: remove amostras da classe majoritária
    - ADASYN: similar ao SMOTE, adaptativo
    - class_weight='balanced': sem resampling, só ajusta pesos
=============================================================================
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constante de configuração
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = 0.5  # Após SMOTE: fraudes = 50% do total de legítimas
                                 # Ex: 200k legítimas → 100k fraudes sintéticas


# ─────────────────────────────────────────────────────────────────────────────
# Função principal
# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTE ao conjunto de treino para balancear as classes.

    ATENÇÃO: SMOTE só deve ser aplicado nos dados de TREINO.
    O conjunto de teste nunca deve ser alterado — ele representa
    o mundo real que o modelo enfrentará em produção.

    Como o SMOTE funciona:
        1. Para cada amostra de fraude real, encontra k vizinhos mais próximos
        2. Cria uma nova amostra sintética no "caminho" entre dois vizinhos
        3. Repete até atingir a proporção desejada (sampling_strategy)

    Args:
        X_train: Features de treino (já normalizadas)
        y_train: Labels de treino

    Returns:
        Tupla (X_resampled, y_resampled) com classes mais balanceadas
    """
    print("\n[⚖] Aplicando SMOTE para balancear as classes no treino...")

    # Distribuição ANTES do SMOTE
    n_fraud_before = y_train.sum()
    n_legit_before = len(y_train) - n_fraud_before
    print(f"    ANTES  → Legítimas: {n_legit_before:>7,} | Fraudes: {n_fraud_before:>5,}")

    # ── Aplica SMOTE ──────────────────────────────────────────────────────────
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
        k_neighbors=5,  # número de vizinhos para gerar amostras sintéticas
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Distribuição DEPOIS do SMOTE
    n_fraud_after = y_resampled.sum()
    n_legit_after = len(y_resampled) - n_fraud_after
    ratio_after = n_legit_after / n_fraud_after

    print(f"    DEPOIS → Legítimas: {n_legit_after:>7,} | Fraudes: {n_fraud_after:>5,}")
    print(f"    Nova razão de desbalanceamento: {ratio_after:.1f}:1")
    print(f"[✓] SMOTE concluído! Total de amostras no treino: {len(X_resampled):,}\n")

    # Converte numpy arrays de volta para DataFrame/Series (melhor compatibilidade)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    return X_resampled, y_resampled
