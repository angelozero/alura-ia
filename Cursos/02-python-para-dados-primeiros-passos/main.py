# %% [markdown]
# # Minha Primeira Célula Interativa
# Este é um exemplo de como usar o Kernel do Jupyter em um arquivo .py

# %%
import pandas as pd
import numpy as np

print("Kernel conectado com sucesso!")

# %%
# Criando um conjunto de dados simples
data = {
    'Nome': ['Alice', 'Bob', 'Charlie'],
    'Pontuação': [85, 92, 78]
}

df = pd.DataFrame(data)

# No modo interativo, basta digitar o nome da variável para visualizar a tabela
df

# %%
# Exemplo de cálculo simples
media = df['Pontuação'].mean()
print(f"A média de pontuação é: {media}")

# %%