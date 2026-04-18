# -*- coding: utf-8 -*-
"""
=============================================================================
main.py — Entry Point do Projeto de Detecção de Fraudes
=============================================================================

Pipeline completo executado em ordem:
    1. 📥 Download automático do dataset (Credit Card Fraud / OpenML)
    2. 🔍 Análise Exploratória de Dados (EDA) + gráficos
    3. ⚙️  Pré-processamento: limpeza, normalização, split treino/teste
    4. ⚖️  Balanceamento com SMOTE (apenas no treino)
    5. 🤖 Treinamento: Logistic Regression, Random Forest, XGBoost
    6. 📊 Avaliação: métricas, confusion matrix, curvas ROC
    7. 💾 Salvamento dos gráficos em outputs/figures/

Como executar:
    $ python main.py

Requisitos:
    $ pip install -r requirements.txt

Saídas:
    • Console: métricas detalhadas de cada modelo
    • outputs/figures/: 6 gráficos salvos em PNG
=============================================================================
"""

import sys
import os
import io

# Força stdout a usar UTF-8 no Windows (evita UnicodeEncodeError no terminal)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time

# ── Adiciona a raiz do projeto ao PYTHONPATH para imports funcionarem ─────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Imports dos módulos do projeto ────────────────────────────────────────────
from src.data.download import download_dataset
from src.data.preprocessing import preprocess
from src.features.engineering import apply_smote
from src.models.train import train_all_models
from src.models.evaluate import evaluate_all_models
from src.visualization.plots import run_eda_plots, run_evaluation_plots


# ─────────────────────────────────────────────────────────────────────────────
# Utilitário: Banner de cabeçalho
# ─────────────────────────────────────────────────────────────────────────────
def print_header() -> None:
    print()
    print("=" * 62)
    print("   DETECCAO DE FRAUDES EM TRANSACOES BANCARIAS")
    print("   Dataset : Credit Card Fraud Detection (OpenML #1597)")
    print("   Modelos : Logistic Regression | Random Forest | XGBoost")
    print("   Tecnica : SMOTE + class_weight para dados desbalanceados")
    print("=" * 62)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Orquestra todo o pipeline de detecção de fraudes.

    Cada etapa é numerada e cronometrada para facilitar o acompanhamento.
    """
    print_header()
    pipeline_start = time.time()

    # ────────────────────────────────────────────────────────────────────────
    # ETAPA 1: Download do Dataset
    # ────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  ETAPA 1/6 — Download do Dataset")
    print("━" * 62)
    df = download_dataset()

    # ────────────────────────────────────────────────────────────────────────
    # ETAPA 2: Análise Exploratória de Dados (EDA)
    # ────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  ETAPA 2/6 — Análise Exploratória de Dados (EDA)")
    print("━" * 62)
    run_eda_plots(df)

    # ────────────────────────────────────────────────────────────────────────
    # ETAPA 3: Pré-processamento
    # ────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  ETAPA 3/6 — Pré-processamento")
    print("━" * 62)
    X_train, X_test, y_train, y_test = preprocess(df)

    # ────────────────────────────────────────────────────────────────────────
    # ETAPA 4: Balanceamento com SMOTE
    # ────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  ETAPA 4/6 — Balanceamento com SMOTE")
    print("━" * 62)
    # SMOTE é aplicado SOMENTE no treino para não contaminar a avaliação
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ────────────────────────────────────────────────────────────────────────
    # ETAPA 5: Treinamento dos Modelos
    # ────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  ETAPA 5/6 — Treinamento dos Modelos")
    print("━" * 62)
    # Treina com os dados balanceados pelo SMOTE
    # O teste permanece com a distribuição original (mundo real)
    trained_models = train_all_models(X_train_bal, y_train_bal)

    # ────────────────────────────────────────────────────────────────────────
    # ETAPA 6: Avaliação e Visualização
    # ────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  ETAPA 6/6 — Avaliação e Visualização")
    print("━" * 62)
    # Avalia no conjunto de TESTE (dados que o modelo nunca viu)
    metrics_df, all_results = evaluate_all_models(trained_models, X_test, y_test)

    # Gera gráficos de avaliação
    run_evaluation_plots(all_results, metrics_df)

    # ────────────────────────────────────────────────────────────────────────
    # Resumo final
    # ────────────────────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 62)
    print(f"  [OK] PIPELINE CONCLUIDO em {total_time:.0f} segundos")
    print()
    print("  Graficos salvos em: outputs/figures/")
    print("    - class_distribution.png")
    print("    - correlation_heatmap.png")
    print("    - amount_by_class.png")
    print("    - confusion_matrices.png")
    print("    - roc_curves.png")
    print("    - metrics_comparison.png")
    print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
