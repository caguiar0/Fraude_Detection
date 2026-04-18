"""
=============================================================================
src/data/download.py
=============================================================================
Modulo responsavel pelo download automatico do dataset de fraudes bancarias.

Dataset: Credit Card Fraud Detection
Fonte:   OpenML (ID 1597) -- sem necessidade de conta Kaggle
Tamanho: ~150 MB | 284.807 transacoes | 492 fraudes (0,17%)

Features:
    - V1 a V28: componentes principais (PCA) -- valores anonimizados
    - Time:     segundos desde a primeira transacao no dataset
    - Amount:   valor da transacao em euros
    - Class:    0 = legitima | 1 = fraudulenta  <- variavel alvo
=============================================================================
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de configuracao
# ─────────────────────────────────────────────────────────────────────────────
OPENML_DATASET_ID = 1597             # ID do dataset no OpenML (deve ser int)
RAW_DATA_DIR = os.path.join("data", "raw")
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "creditcard.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Funcao principal
# ─────────────────────────────────────────────────────────────────────────────
def download_dataset() -> pd.DataFrame:
    """
    Baixa o dataset Credit Card Fraud Detection via OpenML e salva localmente.

    Fluxo:
        1. Verifica se o arquivo ja existe em data/raw/creditcard.csv
        2. Se sim, carrega do disco (evita re-download desnecessario)
        3. Se nao, faz o download via sklearn.datasets.fetch_openml
        4. Salva em CSV para uso posterior

    Returns:
        pd.DataFrame: DataFrame completo com todas as features e o target 'Class'
    """
    # Cria a pasta de dados se nao existir
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Verifica se o CSV ja foi baixado anteriormente
    if os.path.exists(RAW_CSV_PATH):
        print(f"[OK] Dataset encontrado localmente: {RAW_CSV_PATH}")
        print("[->] Carregando do disco (sem re-download)...")
        df = pd.read_csv(RAW_CSV_PATH)
        print(f"[OK] {len(df):,} transacoes carregadas.\n")
        return df

    # Download via OpenML
    print("[..] Dataset nao encontrado. Iniciando download via OpenML...")
    print("     Isso pode levar alguns minutos (~150 MB). Aguarde...\n")

    # fetch_openml retorna features (X) e target (y) separados
    # as_frame=True garante que recebemos DataFrames pandas prontos
    dataset = fetch_openml(
        data_id=OPENML_DATASET_ID,
        as_frame=True,
        parser="auto",
    )

    # Combina features + target num unico DataFrame
    df = dataset.frame  # ja inclui todas as colunas + target 'Class'

    # Garante que 'Class' seja inteiro (0 ou 1)
    df["Class"] = df["Class"].astype(int)

    # Salva em CSV para evitar re-download nas proximas execucoes
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"[OK] Download concluido! Arquivo salvo em: {RAW_CSV_PATH}")
    print(f"[OK] {len(df):,} transacoes | {df['Class'].sum()} fraudes detectadas.\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Execucao direta (teste isolado)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = download_dataset()
    print("Primeiras linhas do dataset:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nDistribuicao das classes:\n{df['Class'].value_counts()}")
