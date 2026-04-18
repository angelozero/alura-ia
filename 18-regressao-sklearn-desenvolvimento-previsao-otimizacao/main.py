# %% [markdown]
# ### Estudo de Predição de Preços de Imóveis - RJ
# Este notebook prepara dados de imóveis e treina um modelo de regressão.
import sys
!{sys.executable} -m pip install yellowbrick

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# %%
# Carregamento dos dados
dados = pd.read_json(path_or_buf="./data/imoveis.json", orient="columns")

# %%
# 1. PREPARAÇÃO E LIMPEZA
# json_normalize expande dicionários aninhados em colunas separadas
dados1 = pd.json_normalize(dados["ident"])
dados2 = pd.json_normalize(dados["listing"], sep="_")

dados_moveis = pd.concat([dados1, dados2], axis=1)

# Filtrando apenas imóveis residenciais no Rio de Janeiro
filtro = (dados_moveis["types_usage"] == "Residencial") & (
    dados_moveis["address_city"] == "Rio de Janeiro"
)
dados_moveis_filtrados = dados_moveis[
    filtro
].copy()  # .copy() evita avisos de SettingWithCopy

# Resetando o índice para evitar problemas na concatenação futura
dados_moveis_filtrados.reset_index(inplace=True, drop=True)

# Tipagem correta: converter strings/objetos para números onde necessário
dados_moveis_filtrados = dados_moveis_filtrados.astype(
    {
        "prices_price": "float64",
        "prices_tax_iptu": "float64",
        "prices_tax_condo": "float64",
        "features_usableAreas": "int64",
        "features_totalAreas": "int64",
    }
)

# Tratamento de valores vazios em Zonas
dados_moveis_filtrados["address_zone"] = dados_moveis_filtrados["address_zone"].replace(
    "", np.nan
)

# Preenchendo Zonas faltantes baseando-se no bairro (se o bairro X é Zona Sul, todos do bairro X serão)
dici = dados_moveis_filtrados[
    ~dados_moveis_filtrados["address_zone"].isna()
].drop_duplicates(subset=["address_neighborhood"])
dic_zonas = {
    row["address_neighborhood"]: row["address_zone"] for _, row in dici.iterrows()
}

for bairro, zona in dic_zonas.items():
    dados_moveis_filtrados.loc[
        dados_moveis_filtrados["address_neighborhood"] == bairro, "address_zone"
    ] = zona

# Preenchendo impostos nulos com 0 (assume-se que não há taxa se não informada)
dados_moveis_filtrados["prices_tax_iptu"].fillna(0, inplace=True)
dados_moveis_filtrados["prices_tax_condo"].fillna(0, inplace=True)

# Removendo colunas que não ajudam na predição (IDs ou constantes)
colunas_para_dropar = [
    "customerID",
    "source",
    "types_usage",
    "address_city",
    "address_location_lon",
    "address_location_lat",
    "address_neighborhood",
]
dados_moveis_filtrados.drop(colunas_para_dropar, axis=1, inplace=True)

# Renomeando para facilitar a manipulação
dicionario_colunas = {
    "types_unit": "unit",
    "address_zone": "zone",
    "prices_price": "price",
    "prices_tax_condo": "tax_condo",
    "prices_tax_iptu": "tax_iptu",
    "features_bedrooms": "bedrooms",
    "features_bathrooms": "bathrooms",
    "features_suites": "suites",
    "features_parkingSpaces": "parkingSpaces",
    "features_usableAreas": "usableAreas",
    "features_totalAreas": "totalAreas",
    "features_floors": "floors",
    "features_unitsOnTheFloor": "unitsOnTheFloor",
    "features_unitFloor": "unitFloor",
}
dados_moveis_filtrados.rename(columns=dicionario_colunas, inplace=True)

# %%
# 2. TRATAMENTO DE ASSIMETRIA
# Preços costumam ter cauda longa (muitos baratos, poucos caríssimos).
# O log ajuda a normalizar essa distribuição para o modelo aprender melhor.
transformer = FunctionTransformer(np.log1p)  # log1p lida com valores zero: log(1+x)

# Selecionamos apenas numéricas para o log
colunas_numericas = dados_moveis_filtrados.select_dtypes(exclude=["object"]).columns
dados_transformados = transformer.transform(dados_moveis_filtrados[colunas_numericas])

# Criamos o DF transformado mantendo as colunas categóricas
df_transformado = pd.concat(
    [
        dados_moveis_filtrados.select_dtypes(include=["object"]),
        pd.DataFrame(dados_transformados, columns=colunas_numericas),
    ],
    axis=1,
)

# Visualizando a nova distribuição do preço
plt.figure(figsize=(10, 5))
sns.histplot(df_transformado["price"], kde=True)
plt.title("Distribuição Logarítmica do Preço")
plt.show()

# %%
# 3. VARIÁVEIS DUMMIES (Categorização)
# Transformar "Zona Sul", "Zona Norte" em colunas 0 e 1
df_final = pd.get_dummies(df_transformado, columns=["unit", "zone"], drop_first=True)

# %%
# 4. MODELO DE MACHINE LEARNING - REGRESSÃO LINEAR
# Descobrindo o preço do imóvel baseado nas características (condominio, caracateristicas do imovel, etc.)

# variáveis explanatórias (independentes) (traz todas as colunas mas sem a coluna de preço, que é o que queremos prever)
X = df_final.drop("price", axis=1)

# variável dependente (com o preço, que é o que queremos prever)
y = df_final["price"]

# divisão em conjunto de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# Fazendo previsões no conjunto de teste
y_pred = modelo.predict(X_teste)

# Avaliando o modelo
r2 = r2_score(y_teste, y_pred)
mae = mean_absolute_error(y_teste, y_pred)
print(f"R²: {r2 * 100:.2f}%")
print(f"MAE: {mae * 100:.2f}%")


# %%
# Desenvolvendo nosso modelo de arvore de decisão
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_treino, y_treino)

DecisionTreeRegressor(max_depth=5, random_state=42)

previsao_dtr = dtr.predict(X_teste)
r2_dtr = r2_score(y_teste, previsao_dtr)
mae_dtr = mean_absolute_error(y_teste, previsao_dtr)

print(f"R² Decision Tree: {r2_dtr * 100:.2f}%")
print(f"MAE Decision Tree: {mae_dtr * 100:.2f}%")

# %%
# Random Forest Regressor
rf = RandomForestRegressor(random_state=42, max_depth=5, n_estimators=10)

rf.fit(X_treino, y_treino)

RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42)

previsao_rf = rf.predict(X_teste)


# %%
