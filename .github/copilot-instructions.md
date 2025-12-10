# Copilot Instructions - Análise de Consumo de Álcool

## Visão Geral do Projeto

Projeto de análise exploratória de dados (EDA) sobre consumo de álcool **per capita** por país usando dados do FiveThirtyEight/WHO de 2010. O projeto é totalmente em **português brasileiro** e organizado como notebook Jupyter educacional com visualizações geográficas interativas.

## Arquitetura e Estrutura

- **Notebook principal**: `alcohol_consumption.ipynb` - fonte única de análise (não há script .py separado)
- **Dados**: `drinks.csv` - 195 países × 5 colunas (country, beer_servings, spirit_servings, wine_servings, total_litres_of_pure_alcohol)
- **Documentação**: `README (2).md` - metadados do dataset (FiveThirtyEight/WHO 2010)

## Convenções Críticas

### 1. Estrutura do Notebook (Rígida)
O notebook segue numeração pedagógica fixa:
1. Importação de Bibliotecas
2. Carregamento dos Dados
3. Reconhecimento e Exploração dos Dados
4. Tratamento e Transformação de Dados
5. Análise Exploratória Visual (EDA Gráfica)

**Ao adicionar conteúdo**: Use subseções numeradas (ex: `### 5.1`, `### 5.2`). Nunca altere a numeração das seções principais.

### 2. Nomenclatura Obrigatória
- DataFrame: `df_drinks` (imutável)
- Coluna de categoria: `consumption_category` (tipo Categorical)
- Labels de categoria: `['Very Low', 'Low', 'Medium', 'High', 'Very High']` (ordem fixa, em inglês por convenção internacional)
- Ordem de categorias: variável `category_order` usada em `color_discrete_sequence` do Plotly

### 3. Sistema de Categorização (Baseado em Quartis)
```python
bins = [0, 1, 4, 7, 10, float('inf')]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
df_drinks['consumption_category'] = pd.cut(df_drinks['total_litres_of_pure_alcohol'], 
                                             bins=bins, labels=labels, include_lowest=True)
```
**Não altere os bins** ([0, 1, 4, 7, 10, ∞]) - são baseados na distribuição quartil dos dados de 2010.

### 4. Stack de Visualização
- **Mapas geográficos**: Plotly Express (`px.choropleth`) com `locationmode='country names'`
- **Estatísticas**: Seaborn + Matplotlib (ainda não implementadas, mas previstas)
- **Importações atuais**:
  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  import plotly.express as px
  ```

## Padrões de Implementação

### Mapas Coropléticos com Plotly
O projeto usa dois tipos de mapa na Seção 5:

**5.1 - Mapa Categórico** (discrete scale):
```python
fig = px.choropleth(
    df_drinks,
    locations="country",
    locationmode='country names',
    color="consumption_category",
    color_discrete_sequence=px.colors.sequential.Reds,
    category_orders={'consumption_category': category_order},  # Ordem fixa importante!
    # ... hover_data com formato ':.2f' para floats
)
fig.update_layout(geo=dict(projection_type='natural earth'), height=600, width=1000)
```

**5.2 - Mapa Contínuo** (continuous scale):
```python
fig2 = px.choropleth(
    df_drinks,
    color="total_litres_of_pure_alcohol",
    color_continuous_scale='Reds',  # Escala contínua, não discrete
    # ... mesmo padrão de hover_data
)
```

**Convenções de hover**:
- Sempre incluir `total_litres_of_pure_alcohol` com formato `':.2f'`
- Sempre incluir as três colunas de servings (beer, wine, spirit)
- Ocultar `category_numeric` se criado (`False` no hover_data)

### Transformações de Dados (Seção 4)
- **Sempre** preceder código com célula markdown explicando a lógica/critérios
- **Sempre** verificar resultado com `.value_counts()` ou `.head()` em célula separada
- Exemplo: Seção 4.1 cria a categoria, 4.2 verifica com `value_counts().sort_index()`

### Documentação em Markdown
- **Títulos de seção**: Use `##` para seções principais, `###` para subseções
- **Ênfase em per capita**: Sempre mencionar "por pessoa" ou "per capita" ao descrever métricas
- **Fontes**: Incluir citação completa (FiveThirtyEight + WHO + ano) em células introdutórias

## Notas sobre os Dados

- **195 países** (2010) sem valores nulos
- **Unidades**: servings (doses por pessoa/ano), total_litres (litros de álcool puro por pessoa/ano)
- **Países com consumo zero**: Afghanistan, Bangladesh (contexto religioso/cultural)
- **Coluna country**: Usa nomes completos compatíveis com `plotly`'s `locationmode='country names'` (ex: "Antigua & Barbuda")

## Workflows Comuns

**Adicionar nova visualização geográfica**:
1. Criar subseção em `## 5` (ex: `### 5.3`)
2. Documentar o propósito/insight em markdown
3. Seguir padrão de `px.choropleth` com `locationmode='country names'`
4. Usar `color_discrete_sequence` ou `color_continuous_scale` com paleta `Reds` (consistência visual)

**Adicionar análise estatística**:
1. Inserir em nova subseção de `## 5` (EDA Gráfica) ou `## 3` (Exploração)
2. Preferir `seaborn` para gráficos estatísticos (histogramas, boxplots, correlações)
3. Usar configuração padrão: `sns.set_style()` se necessário customizar tema
