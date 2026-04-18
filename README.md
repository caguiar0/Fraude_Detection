# 🔍 Detecção de Fraudes em Transações Bancárias

Projeto educacional em Python para detecção de fraudes usando Machine Learning.

## 📁 Estrutura do Projeto

```
fraud_detection/
├── data/
│   └── raw/                  # Dataset original (baixado automaticamente)
├── outputs/
│   └── figures/              # Gráficos gerados
├── src/
│   ├── data/
│   │   ├── download.py       # Download automático do dataset
│   │   └── preprocessing.py  # Pré-processamento e split
│   ├── features/
│   │   └── engineering.py    # Feature engineering e balanceamento
│   ├── models/
│   │   ├── train.py          # Treinamento dos modelos
│   │   └── evaluate.py       # Avaliação e métricas
│   └── visualization/
│       └── plots.py          # Visualizações EDA e resultados
├── main.py                   # 🚀 Entry point — rode este arquivo
└── requirements.txt
```

## 🚀 Como Usar

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Execute o pipeline completo
```bash
python main.py
```

O script irá:
1. 📥 Baixar o dataset automaticamente (Credit Card Fraud, ~150 MB)
2. 🔍 Fazer análise exploratória e salvar gráficos em `outputs/figures/`
3. ⚙️  Pré-processar os dados e aplicar SMOTE para balanceamento
4. 🤖 Treinar 3 modelos: Logistic Regression, Random Forest e XGBoost
5. 📊 Exibir e salvar métricas detalhadas de avaliação

## 📊 Dataset

**Credit Card Fraud Detection** (ULB Machine Learning Group)
- 284.807 transações europeias de cartão de crédito (setembro de 2013)
- 492 transações fraudulentas (~0,17% — altamente desbalanceado)
- Features V1–V28: resultado de PCA para anonimização
- Features originais: `Time`, `Amount`, `Class` (0=Normal, 1=Fraude)

Fonte: [OpenML #1597](https://www.openml.org/d/1597)

## 🤖 Modelos

| Modelo               | Por que usar?                                      |
|----------------------|----------------------------------------------------|
| Logistic Regression  | Baseline simples, interpretável, rápido            |
| Random Forest        | Robusto, lida bem com dados tabulares              |
| XGBoost              | Estado da arte para tabular, excelente com imbalance |

## ⚖️ Desbalanceamento de Classes

O dataset é **extremamente desbalanceado** (~0,17% fraudes). Estratégias usadas:
- **SMOTE**: gera amostras sintéticas da classe minoritária no treino
- **`class_weight='balanced'`**: penaliza mais os erros na classe minoritária
- **Métricas adequadas**: ROC-AUC, F1-Score, Precision, Recall (evitar accuracy simples!)

## 📈 Outputs

Após rodar `main.py`, verifique a pasta `outputs/figures/` para:
- Distribuição das classes
- Heatmap de correlações
- Distribuição do valor das transações (fraude vs normal)
- Confusion Matrix de cada modelo
- Curvas ROC comparativas
- Comparativo de métricas entre modelos
