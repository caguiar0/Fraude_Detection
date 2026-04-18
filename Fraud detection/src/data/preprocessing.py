"""
=============================================================================
src/data/preprocessing.py
=============================================================================
Módulo de pré-processamento dos dados de fraude.

Responsabilidades:
    1. Inspecionar e validar a qualidade dos dados (valores nulos, tipos)
    2. Normalizar as features 'Time' e 'Amount' (V1-V28 já foram normalizadas via PCA)
    3. Separar features (X) e variável alvo (y)
    4. Dividir em conjuntos de treino e teste com estratificação
       (mantém a proporção de fraudes em treino e teste)

Por que normalizar 'Time' e 'Amount'?
    - V1–V28 já vieram normalizadas via PCA no dataset original
    - 'Time' e 'Amount' estão em escalas muito diferentes (segundos vs euros)
    - Algoritmos como Logistic Regression são sensíveis à escala das features
    - StandardScaler: transforma para média=0 e desvio padrão=1
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de configuração
# ─────────────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Class"          # Coluna alvo: 0=legítima, 1=fraude
TEST_SIZE = 0.20                 # 20% dos dados para teste
RANDOM_STATE = 42                # Semente para reprodutibilidade
# Colunas que podem precisar de normalização (V1-V28 já estão normalizadas via PCA)
# Verificamos dinamicamente quais existem no dataset
POSSIBLE_SCALE_COLS = ["Time", "Amount"]  # depende da fonte do dataset


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """
    Exibe um relatório resumido sobre o dataset:
    shape, tipos, valores nulos, e distribuição das classes.
    """
    print("=" * 60)
    print("  INSPEÇÃO DO DATASET")
    print("=" * 60)
    print(f"  Dimensões:            {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"  Valores nulos:        {df.isnull().sum().sum()}")
    print(f"  Duplicatas:           {df.duplicated().sum():,}")
    print()

    # Distribuição das classes com percentuais
    counts = df[TARGET_COLUMN].value_counts()
    total = len(df)
    print("  Distribuição das classes:")
    print(f"    Legítimas  (0): {counts[0]:>7,}  ({counts[0]/total*100:.2f}%)")
    print(f"    Fraudes    (1): {counts[1]:>7,}  ({counts[1]/total*100:.2f}%)")
    print(f"    Razão de desbalanceamento: {counts[0]/counts[1]:.0f}:1")
    print("=" * 60)
    print()


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normaliza as features não-PCA (ex: 'Time', 'Amount') usando StandardScaler.

    IMPORTANTE: o scaler é FITADO apenas no conjunto de treino e depois
    aplicado (transform) no teste. Isso evita data leakage — o modelo
    nunca "vê" estatísticas do conjunto de teste durante o treino.

    As colunas V1-V28 já vieram normalizadas via PCA no dataset original.
    Detectamos dinamicamente quais colunas precisam de escala.

    Args:
        X_train: Features de treino
        X_test:  Features de teste

    Returns:
        Tupla (X_train_scaled, X_test_scaled) com colunas normalizadas
    """
    # Detecta quais colunas candidatas realmente existem no dataset
    cols_to_scale = [c for c in POSSIBLE_SCALE_COLS if c in X_train.columns]

    if not cols_to_scale:
        print("[!] Nenhuma coluna para normalizar encontrada. Pulando escala.")
        return X_train, X_test

    scaler = StandardScaler()

    # Cria cópias para não modificar os DataFrames originais
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit no treino → transform em ambos (evita data leakage)
    X_train_scaled[cols_to_scale] = scaler.fit_transform(
        X_train[cols_to_scale]
    )
    X_test_scaled[cols_to_scale] = scaler.transform(
        X_test[cols_to_scale]
    )

    print(f"[✓] Features normalizadas: {cols_to_scale}")
    ref_col = cols_to_scale[-1]  # usa a última coluna encontrada como referência
    print(f"    Média após escala  (treino): {X_train_scaled[ref_col].mean():.4f}")
    print(f"    Desvio após escala (treino): {X_train_scaled[ref_col].std():.4f}\n")

    return X_train_scaled, X_test_scaled


# ─────────────────────────────────────────────────────────────────────────────
# Função principal
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline completo de pré-processamento.

    Etapas:
        1. Inspeciona o dataset
        2. Remove duplicatas (se houver)
        3. Separa X (features) e y (target)
        4. Divide em treino/teste estratificado
        5. Normaliza 'Time' e 'Amount'

    Args:
        df: DataFrame bruto carregado pelo módulo download.py

    Returns:
        Tupla: (X_train, X_test, y_train, y_test)
    """
    print("\n[⚙] Iniciando pré-processamento...\n")

    # ── 1. Inspeção ───────────────────────────────────────────────────────────
    inspect_data(df)

    # ── 2. Remove duplicatas (dataset tem algumas linhas duplicadas) ───────────
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"[!] {n_removed} duplicatas removidas.")

    # ── 3. Separação X / y ────────────────────────────────────────────────────
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"[✓] Features (X): {X.shape[1]} colunas")
    print(f"[✓] Target  (y): '{TARGET_COLUMN}' — {y.sum()} fraudes em {len(y):,} amostras\n")

    # ── 4. Divisão treino/teste com estratificação ────────────────────────────
    # stratify=y garante que a proporção de fraudes é mantida em ambos os splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # ← ESSENCIAL para dados desbalanceados
    )

    print(f"[✓] Split treino/teste ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}):")
    print(f"    Treino: {len(X_train):>7,} amostras | {y_train.sum():>4} fraudes")
    print(f"    Teste:  {len(X_test):>7,} amostras | {y_test.sum():>4} fraudes\n")

    # ── 5. Normalização ───────────────────────────────────────────────────────
    X_train, X_test = scale_features(X_train, X_test)

    print("[✓] Pré-processamento concluído!\n")
    return X_train, X_test, y_train, y_test
