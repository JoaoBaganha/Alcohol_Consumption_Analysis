# 🍺 Alcohol Consumption Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

*Análise exploratória e modelagem preditiva do consumo de álcool per capita em 195 países*

[🔍 Explorar Notebook](alcohol_consumption.ipynb) • [📊 Visualizar Dados](drinks.csv) • [📝 Relatório Completo](#resultados)

</div>

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Dados](#-dados)
- [Metodologia](#-metodologia)
- [Principais Descobertas](#-principais-descobertas)
- [Modelagem Preditiva](#-modelagem-preditiva)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Como Executar](#-como-executar)
- [Resultados](#-resultados)
- [Contribuidores](#-contribuidores)
- [Licença](#-licença)

---

## 🎯 Visão Geral

Este projeto realiza uma **análise exploratória de dados (EDA)** e **modelagem preditiva** do consumo de álcool per capita em 195 países, utilizando dados de 2010 da Organização Mundial da Saúde (OMS). O objetivo é identificar padrões geográficos, culturais e estatísticos que explicam o consumo de álcool globalmente.

### Objetivos Principais

1. **Explorar padrões globais** de consumo de álcool per capita
2. **Testar hipóteses** sobre fatores culturais e tipos de bebida
3. **Desenvolver modelos preditivos** (regressão e classificação)
4. **Fornecer insights** para políticas públicas de saúde

---

## 📊 Dados

### Fonte

- **FiveThirtyEight** (2014): ["Dear Mona Followup: Where Do People Drink The Most Beer, Wine And Spirits?"](https://fivethirtyeight.com/features/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/)
- **World Health Organisation (WHO)**: Global Information System on Alcohol and Health (GISAH), 2010

### Variáveis

| Variável | Descrição | Unidade |
|----------|-----------|---------|
| `country` | Nome do país | - |
| `beer_servings` | Doses médias de cerveja por pessoa/ano | doses |
| `spirit_servings` | Doses médias de destilados por pessoa/ano | doses |
| `wine_servings` | Doses médias de vinho por pessoa/ano | doses |
| `total_litres_of_pure_alcohol` | Litros de álcool puro por pessoa/ano | litros |

### Características

- **195 países** analisados
- **Sem valores nulos** ou duplicados
- **Outliers mantidos** (valores reais de países com alto consumo)
- **Granularidade**: Dados agregados por país (2010)

---

## 🔬 Metodologia

### 1. Análise Exploratória de Dados (EDA)

- Estatísticas descritivas e distribuições
- Identificação e tratamento de outliers
- Análise de correlações entre variáveis
- Visualizações geográficas interativas (mapas-múndi)
- Categorização de países por nível de consumo

### 2. Testes de Hipóteses

#### **Hipótese I**: Países islâmicos consomem menos álcool?
- **Teste**: Welch's t-test (variâncias desiguais)
- **Resultado**: ✅ **Confirmada** (p < 0.001)
- **Diferença**: ~5.6 litros/ano entre grupos

#### **Hipótese II**: Qual bebida explica melhor o consumo total?
- **Análise**: Correlações de Pearson
- **Resultado**: ✅ **Beer (r=0.83)** > Wine (r=0.66) > Spirits (r=0.65)

### 3. Modelagem Preditiva

#### **Regressão** (predizer litros de álcool)
- Linear Múltipla
- Ridge (regularização L2)
- Polinomial (grau 2)

#### **Classificação** (predizer alto consumo)
- Logistic Regression
- Gaussian Naive Bayes
- Tuning de hiperparâmetros (RandomizedSearchCV)

### 4. Validação

- **Validação cruzada** (k=5)
- **Análise de resíduos**
- **VIF** (multicolinearidade)
- **Intervalos de confiança** (95%)

---

## 🔑 Principais Descobertas

### 📍 Padrões Geográficos

- **Alto consumo**: Europa Central/Oriental, Rússia
- **Baixo consumo**: Oriente Médio, Norte da África, Sudeste Asiático
- **Fatores**: Cultura, religião, clima, políticas públicas

### 🕌 Influência Cultural

- Países de maioria islâmica consomem **~5.6 litros/ano a menos** (diferença estatisticamente significativa)
- Religião é um **forte preditor** de padrões de consumo

### 🍺 Tipos de Bebida

| Bebida | Correlação com Consumo Total | Insight |
|--------|------------------------------|---------|
| **Beer** | **r=0.83** | **Principal contribuidor** |
| Wine | r=0.66 | Relevante em Europa Ocidental |
| Spirits | r=0.65 | Forte em Europa Oriental/Rússia |

**Implicação**: Políticas de controle do consumo de cerveja teriam **maior impacto** na saúde pública.

---

## 🤖 Modelagem Preditiva

### Resultados: Regressão

| Modelo | R² Ajustado | RMSE (Litros) | Complexidade |
|--------|-------------|---------------|--------------|
| **Linear Múltipla** | **0.91** | **1.2** | ⭐ Simples |
| Ridge (tunado) | 0.90 | 1.1 | ⭐⭐ Moderada |
| Polinomial (grau 2) | 0.92 | 1.0 | ⭐⭐⭐ Alta |

**Recomendação**: **Linear Múltipla** (melhor custo-benefício)

### Resultados: Classificação

| Modelo | Acurácia | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Gaussian NB | 0.82 | 0.78 | 0.83 |
| **Logistic Regression** | **0.88** | **0.85** | **0.88** |
| Logistic (tunado) | 0.89 | 0.87 | 0.89 |

**Recomendação**: **Logistic Regression** (base ou tunado)

### Qualidade dos Modelos

✅ **VIF < 3** → Sem multicolinearidade severa  
✅ **Resíduos centrados** → Pressupostos atendidos  
✅ **p-valores < 0.001** → Coeficientes significativos  
✅ **Validação cruzada** → Boa generalização  

---

## 🛠️ Tecnologias Utilizadas

### Linguagem e Ambiente
- **Python 3.8+**
- **Jupyter Notebook / JupyterLab**

### Bibliotecas

```python
# Manipulação e análise de dados
pandas, numpy

# Visualização
matplotlib, seaborn, plotly

# Modelagem e estatística
scikit-learn, statsmodels, scipy
```

### Principais Técnicas

- **EDA**: Mapas interativos (Plotly), scatter plots, heatmaps
- **Estatística**: Testes t, correlações, intervalos de confiança
- **Machine Learning**: Regressão, classificação, tuning, validação cruzada

---

## 🚀 Como Executar

### 1. Clone o Repositório

```bash
git clone https://github.com/JoaoBaganha/Alcohol_Consumption_Analysis.git
cd Alcohol_Consumption_Analysis
```

### 2. Instale as Dependências

```bash
pip install pandas seaborn matplotlib plotly numpy scipy scikit-learn statsmodels jupyterlab
```

### 3. Execute o Notebook

```bash
jupyter lab alcohol_consumption.ipynb
```

### 4. Explore os Resultados

- Execute as células sequencialmente
- Interaja com os mapas e gráficos
- Ajuste parâmetros dos modelos (opcional)

---

## 📈 Resultados

### Síntese Final

| Aspecto | Resultado |
|---------|-----------|
| **Dataset** | 195 países, dados limpos, outliers mantidos |
| **EDA** | Padrões geográficos e culturais claros |
| **Hipóteses** | Ambas confirmadas (p < 0.05) |
| **Melhor Modelo (Regressão)** | Linear Múltipla (R²=0.91) |
| **Melhor Modelo (Classificação)** | Logistic Regression (F1=0.85) |
| **Insight Principal** | **Beer é o principal preditor** do consumo total |

### Trade-offs e Recomendações

**Para fins educacionais/exploratórios**:
- Modelos simples e interpretáveis (Linear Múltipla, Logistic Regression)

**Para produção com mais dados**:
- Ridge/Lasso com tuning robusto
- Ensemble methods (Random Forest, XGBoost)

---

## 👥 Contribuidores

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/JoaoBaganha">
        <img src="https://github.com/JoaoBaganha.png" width="100px;" alt="João Baganha"/><br>
        <sub><b>João Baganha</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/PedroCarneiro">
        <img src="https://github.com/PedroCarneiro.png" width="100px;" alt="Pedro Carneiro"/><br>
        <sub><b>Pedro Carneiro</b></sub>
      </a>
    </td>
  </tr>
</table>

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

### Licenças dos Dados

- **Dataset**: FiveThirtyEight (MIT-compatible)
- **Uso**: Educacional e não-comercial

---

## 📞 Contato

- **GitHub**: [@JoaoBaganha](https://github.com/JoaoBaganha)
- **Email**: [Seu email aqui]
- **LinkedIn**: [Seu LinkedIn aqui]

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela!**

*Última atualização: 11 de dezembro de 2025*

</div>
