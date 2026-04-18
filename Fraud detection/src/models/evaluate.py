"""
=============================================================================
src/models/evaluate.py
=============================================================================
Módulo de avaliação dos modelos de detecção de fraude.

Por que NÃO usar Accuracy como métrica principal?
    Com 0,17% de fraudes, um modelo que "chuta" sempre LEGÍTIMO acerta
    99,83% das vezes — mas detecta ZERO fraude! Isso é perigoso.

Métricas corretas para dados desbalanceados:
    ┌─────────────────┬─────────────────────────────────────────────────┐
    │ Métrica         │ O que mede                                      │
    ├─────────────────┼─────────────────────────────────────────────────┤
    │ Precision       │ Dos alertados como fraude, quantos são reais?   │
    │                 │ Alta precision = poucos falsos alarmes           │
    ├─────────────────┼─────────────────────────────────────────────────┤
    │ Recall          │ Das fraudes reais, quantas foram detectadas?     │
    │ (Sensitivity)   │ Alto recall = poucos fraudadores escapam        │
    ├─────────────────┼─────────────────────────────────────────────────┤
    │ F1-Score        │ Média harmônica entre Precision e Recall        │
    │                 │ Equilíbrio entre os dois                        │
    ├─────────────────┼─────────────────────────────────────────────────┤
    │ ROC-AUC         │ Área sob a curva ROC — capacidade discriminativa│
    │                 │ 1.0 = perfeito | 0.5 = aleatório               │
    ├─────────────────┼─────────────────────────────────────────────────┤
    │ PR-AUC          │ Área sob Precision-Recall — melhor para         │
    │                 │ datasets muito desbalanceados                    │
    └─────────────────┴─────────────────────────────────────────────────┘

No contexto de fraude bancária, o trade-off mais importante é:
    • Recall alto → menos fraudes escapam (cliente protegido)
    • Precision alta → menos transações legítimas bloqueadas (cliente satisfeito)
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Função de avaliação individual
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Avalia um modelo individual e exibe relatório detalhado no console.

    Args:
        model_name: Nome amigável do modelo
        model:      Estimator treinado (sklearn API)
        X_test:     Features de teste
        y_test:     Labels reais de teste

    Returns:
        dict com as métricas calculadas
    """
    print(f"\n{'─' * 60}")
    print(f"  Avaliando: {model_name}")
    print(f"{'─' * 60}")

    # ── Predições ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)                # Classes preditas (0 ou 1)
    y_proba = model.predict_proba(X_test)[:, 1]   # Probabilidade de ser fraude

    # ── Métricas principais ───────────────────────────────────────────────────
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)
    f1      = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── Exibição no console ───────────────────────────────────────────────────
    print(f"\n  Confusion Matrix:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │              Predito como:              │")
    print(f"  │           Legítima    Fraude            │")
    print(f"  │ Real Leg. │ TN={tn:>5,}  │ FP={fp:>4,}  │ ← Falsos alarmes")
    print(f"  │ Real Fra. │ FN={fn:>5,}  │ TP={tp:>4,}  │ ← Fraudes detectadas")
    print(f"  └─────────────────────────────────────────┘")
    print(f"\n  FN={fn} → fraudes NÃO detectadas (risco alto!)")
    print(f"  FP={fp} → legítimas bloqueadas erroneamente")

    print(f"\n  Métricas de Classificação:")
    print(f"  {'Metric':<20} {'Valor':>10}")
    print(f"  {'─'*32}")
    print(f"  {'Precision':<20} {precision:>10.4f}")
    print(f"  {'Recall (Sensitivity)':<20} {recall:>10.4f}")
    print(f"  {'F1-Score':<20} {f1:>10.4f}")
    print(f"  {'ROC-AUC':<20} {roc_auc:>10.4f}")
    print(f"  {'PR-AUC':<20} {pr_auc:>10.4f}")

    print(f"\n  Relatório Completo (sklearn):")
    print(classification_report(
        y_test, y_pred,
        target_names=["Legítima (0)", "Fraude (1)"],
        digits=4,
    ))

    # ── Retorna métricas para comparação posterior ────────────────────────────
    return {
        "Model":     model_name,
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1-Score":  round(f1, 4),
        "ROC-AUC":   round(roc_auc, 4),
        "PR-AUC":    round(pr_auc, 4),
        "TP":        int(tp),
        "FP":        int(fp),
        "FN":        int(fn),
        "TN":        int(tn),
        # Armazena dados para plotagem das curvas ROC
        "_y_test":   y_test,
        "_y_proba":  y_proba,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Função de avaliação de todos os modelos
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Avalia todos os modelos treinados e retorna um DataFrame comparativo.

    Args:
        trained_models: dict { nome_modelo: estimator_treinado }
        X_test:         Features de teste
        y_test:         Labels de teste

    Returns:
        Tupla:
            - pd.DataFrame com métricas comparativas de todos os modelos
            - dict com resultados completos (incluindo dados para plotagem)
    """
    print("\n" + "=" * 60)
    print("  AVALIAÇÃO DOS MODELOS")
    print("=" * 60)

    all_results = {}
    metrics_list = []

    for name, model in trained_models.items():
        result = evaluate_model(name, model, X_test, y_test)
        all_results[name] = result

        # Coleta apenas métricas numéricas (sem os arrays de dados)
        metrics_list.append({
            k: v for k, v in result.items()
            if not k.startswith("_")
        })

    # ── Tabela comparativa ────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_list).set_index("Model")

    print("\n" + "=" * 60)
    print("  COMPARATIVO FINAL DE MÉTRICAS")
    print("=" * 60)
    print(metrics_df[["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]].to_string())
    print()

    # Destaca o melhor modelo em cada métrica
    best_by_metric = metrics_df[["F1-Score", "ROC-AUC", "PR-AUC"]].idxmax()
    print("  🏆 Melhor modelo por métrica:")
    for metric, best_model in best_by_metric.items():
        print(f"     {metric:<10}: {best_model}")

    return metrics_df, all_results
