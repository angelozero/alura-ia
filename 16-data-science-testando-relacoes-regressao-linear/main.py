# Regressão Linear
# %%
import sys
!{sys.executable} -m pip install plotly statsmodels

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols

# %%
dados = pd.read_csv("./data/Preços_de_casas.csv")


# %%
dados.head()
dados = dados.drop(columns="Id")
correlacao = dados.corr()
print(correlacao["preco_de_venda"].sort_values(ascending=False))


# %%
# Grafico de dispersão
plt.scatter(dados["area_primeiro_andar"], dados["preco_de_venda"])
plt.xlabel("Área do Primeiro Andar")
plt.ylabel("Preço de Venda")
plt.title("Relação entre Área e Preço de Venda")
plt.show()

# %%
plt.scatter(dados["area_primeiro_andar"], dados["preco_de_venda"])
plt.axline( xy1 = (66, 250000), color="red", xy2=(190, 1800000), linestyle="--", label="Média da Área do Primeiro Andar")
plt.legend()
plt.xlabel("Área do Primeiro Andar")
plt.ylabel("Preço de Venda")
plt.title("Relação entre Área e Preço de Venda")
plt.show()

# %%
px.scatter(dados, x="area_primeiro_andar", y="preco_de_venda", trendline_color_override="red", trendline="ols", title="Relação entre Área e Preço de Venda")

# %%
sns.displot(dados["preco_de_venda"], kde=True, color="green")
plt.title("Distribuição do Preço de Venda")
plt.show()

# %%
# Treinando o modelo de regressão linear
y = dados["preco_de_venda"]
X = dados.drop(columns="preco_de_venda")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=230)

# %% 
# Base de treino
df_train = pd.DataFrame(X_train)
df_train["preco_de_venda"] = y_train
modelo_0 = ols("preco_de_venda ~ area_primeiro_andar", data=df_train).fit()
print(modelo_0.params)
 
# %%
modelo_0.rsquared
modelo_0.resid.hist()
plt.title("Distribuição dosResíduos do Modelo de Regressão Linear")
plt.show()

# %%
y_pred = modelo_0.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² do modelo: {r2:.4f}")

# %%
sns.pairplot(dados, x_vars=["area_primeiro_andar"], y_vars="preco_de_venda", height=5, aspect=1.5, kind="reg")
# %%
