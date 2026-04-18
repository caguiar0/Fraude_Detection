"""
=============================================================================
src/visualization/plots.py
=============================================================================
Módulo de visualizações para análise exploratória e avaliação dos modelos.

Gráficos gerados:
    EDA (Análise Exploratória):
        1. class_distribution.png     — Distribuição das classes (fraude vs normal)
        2. correlation_heatmap.png    — Heatmap de correlação das features
        3. amount_by_class.png        — Distribuição do valor das transações

    Avaliação dos Modelos:
        4. confusion_matrices.png     — Confusion matrix de cada modelo
        5. roc_curves.png             — Curvas ROC comparativas
        6. metrics_comparison.png     — Barplot comparando F1, ROC-AUC, PR-AUC

Todos os gráficos são salvos em outputs/figures/
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Configuração visual global
# ─────────────────────────────────────────────────────────────────────────────
# Paleta de cores do projeto
PALETTE = {
    "fraud":    "#E74C3C",   # Vermelho — fraude
    "legit":    "#2ECC71",   # Verde    — legítimo
    "primary":  "#3498DB",   # Azul     — destaque
    "dark":     "#1A1A2E",   # Fundo escuro
    "gray":     "#7F8C8D",   # Cinza    — neutro
}

# Cores distintas para cada modelo
MODEL_COLORS = ["#3498DB", "#E67E22", "#9B59B6"]

FIGURES_DIR = os.path.join("outputs", "figures")

# Aplica estilo visual consistente
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Utilitário interno
# ─────────────────────────────────────────────────────────────────────────────
def _save_figure(filename: str) -> None:
    """Salva a figura atual em outputs/figures/ e fecha o plot."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[📊] Gráfico salvo: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos de EDA
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Gráfico de barras mostrando o desbalanceamento extremo entre classes.

    Este gráfico é fundamental para comunicar o desafio central do problema:
    a classe de fraude é ~0,17% do total, tornando métricas simples inúteis.
    """
    counts = df["Class"].value_counts().sort_index()
    labels = ["Legítima (0)", "Fraude (1)"]
    colors = [PALETTE["legit"], PALETTE["fraud"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição das Classes: Legítima vs Fraude", fontsize=16, fontweight="bold", y=1.02)

    # ── Gráfico 1: Contagem absoluta ──────────────────────────────────────────
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    axes[0].set_title("Contagem Absoluta", fontsize=13)
    axes[0].set_ylabel("Número de Transações")

    # Adiciona valores nas barras
    for bar, count in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f"{count:,}",
            ha="center", va="bottom", fontweight="bold", fontsize=11
        )

    # ── Gráfico 2: Pizza com percentuais ──────────────────────────────────────
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.3f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    axes[1].set_title("Proporção Percentual", fontsize=13)

    plt.tight_layout()
    _save_figure("class_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap de correlação entre as features V1–V28, Time e Amount.

    Útil para identificar features altamente correlacionadas e entender
    a estrutura dos dados após PCA.
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    # Calcula correlação de Pearson
    corr = df.drop(columns=["Class"]).corr()

    # Máscara para o triângulo superior (evita duplicação)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlação de Pearson"},
        annot=False,   # False pois há muitas features (dificulta leitura)
    )

    ax.set_title("Heatmap de Correlação entre Features", fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    _save_figure("correlation_heatmap.png")


def plot_amount_by_class(df: pd.DataFrame) -> None:
    """
    Compara a distribuição do valor das transações entre classes.

    Análise importante: transações fraudulentas tendem a ter valores
    menores (criminosos testam pequenos valores antes de grandes fraudes)
    ou muito específicos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribuição do Valor das Transações por Classe", fontsize=16, fontweight="bold")

    fraud = df[df["Class"] == 1]["Amount"]
    legit = df[df["Class"] == 0]["Amount"]

    # ── Histograma com scale log ───────────────────────────────────────────────
    axes[0].hist(legit, bins=60, color=PALETTE["legit"], alpha=0.7, label=f"Legítima (n={len(legit):,})")
    axes[0].hist(fraud, bins=60, color=PALETTE["fraud"], alpha=0.8, label=f"Fraude (n={len(fraud):,})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Valor da Transação (€)")
    axes[0].set_ylabel("Contagem (escala log)")
    axes[0].set_title("Histograma de Valores")
    axes[0].legend()

    # ── Boxplot comparativo ────────────────────────────────────────────────────
    data_to_plot = [legit, fraud]
    bp = axes[1].boxplot(
        data_to_plot,
        patch_artist=True,
        labels=["Legítima (0)", "Fraude (1)"],
        notch=True,
    )
    colors_box = [PALETTE["legit"], PALETTE["fraud"]]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_ylabel("Valor da Transação (€)")
    axes[1].set_title("Boxplot — Mediana e Dispersão")

    # Adiciona stats
    for i, data in enumerate([legit, fraud], 1):
        axes[1].text(
            i, data.max() * 0.9,
            f"Média: €{data.mean():.0f}\nMediana: €{data.median():.0f}",
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    plt.tight_layout()
    _save_figure("amount_by_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos de Avaliação dos Modelos
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(all_results: Dict[str, Dict]) -> None:
    """
    Plota a confusion matrix de cada modelo lado a lado.

    A confusion matrix mostra:
        TN | FP      Verdadeiros Negativos | Falsos Positivos
        FN | TP      Falsos Negativos      | Verdadeiros Positivos

    Para fraude: FN é o erro mais crítico (fraude não detectada)!
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    fig.suptitle("Confusion Matrix por Modelo", fontsize=16, fontweight="bold")

    if n_models == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, all_results.items()):
        tn = result["TN"]
        fp = result["FP"]
        fn = result["FN"]
        tp = result["TP"]

        cm = np.array([[tn, fp], [fn, tp]])

        # Normaliza para percentual (mais legível)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=13, fontweight="bold")

        classes = ["Legítima (0)", "Fraude (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10, rotation=90, va="center")
        ax.set_xlabel("Predito", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)

        # Adiciona valores com percentual
        labels = [
            [f"TN\n{tn:,}\n({cm_pct[0,0]:.1f}%)", f"FP\n{fp:,}\n({cm_pct[0,1]:.1f}%)"],
            [f"FN\n{fn:,}\n({cm_pct[1,0]:.1f}%)", f"TP\n{tp:,}\n({cm_pct[1,1]:.1f}%)"],
        ]
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    _save_figure("confusion_matrices.png")


def plot_roc_curves(all_results: Dict[str, Dict]) -> None:
    """
    Curvas ROC (Receiver Operating Characteristic) de todos os modelos.

    A curva ROC plota TPR vs FPR em diferentes limiares de decisão.
    ROC-AUC = 1.0: modelo perfeito | ROC-AUC = 0.5: aleatório (diagonal)

    Para dados MUITO desbalanceados, prefer ROC-AUC mas também olhar PR-AUC.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for (name, result), color in zip(all_results.items(), MODEL_COLORS):
        y_test  = result["_y_test"]
        y_proba = result["_y_proba"]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {roc_auc:.4f})")

    # Linha de referência (classificador aleatório)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Aleatório (AUC = 0.50)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=13)
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=13)
    ax.set_title("Curvas ROC — Comparativo de Modelos", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)

    # Área de interesse: canto superior esquerdo (alta TPR, baixo FPR)
    ax.fill_between([0, 0.1], [0, 0], [1, 1], alpha=0.05, color="green",
                    label="_nolegend_")
    ax.text(0.05, 0.5, "↑\nZona\nIdeal", ha="center", va="center",
            fontsize=9, color="green", alpha=0.6)

    plt.tight_layout()
    _save_figure("roc_curves.png")


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    """
    Gráfico de barras comparando as principais métricas entre modelos.

    Permite uma visão rápida de qual modelo performa melhor em cada aspecto.
    """
    metrics_to_plot = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    plot_df = metrics_df[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Barras agrupadas
    n_metrics = len(metrics_to_plot)
    n_models = len(plot_df)
    x = np.arange(n_metrics)
    width = 0.25

    for i, (model_name, row) in enumerate(plot_df.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            row.values,
            width=width * 0.9,
            color=MODEL_COLORS[i],
            alpha=0.85,
            label=model_name,
            edgecolor="white",
        )
        # Valores nas barras
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Comparativo de Métricas por Modelo", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    _save_figure("metrics_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Função orquestradora — EDA
# ─────────────────────────────────────────────────────────────────────────────
def run_eda_plots(df: pd.DataFrame) -> None:
    """
    Executa todos os gráficos de análise exploratória.

    Args:
        df: DataFrame bruto com o dataset completo
    """
    print("\n[📊] Gerando gráficos de Análise Exploratória (EDA)...")
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_amount_by_class(df)
    print("[✓] Gráficos de EDA concluídos!\n")


# ─────────────────────────────────────────────────────────────────────────────
# Função orquestradora — Avaliação
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation_plots(
    all_results: Dict[str, Dict],
    metrics_df: pd.DataFrame,
) -> None:
    """
    Executa todos os gráficos de avaliação dos modelos.

    Args:
        all_results: dict com resultados completos de cada modelo
        metrics_df:  DataFrame com métricas numéricas (para comparação)
    """
    print("\n[📊] Gerando gráficos de avaliação dos modelos...")
    plot_confusion_matrices(all_results)
    plot_roc_curves(all_results)
    plot_metrics_comparison(metrics_df)
    print("[✓] Gráficos de avaliação concluídos!\n")
