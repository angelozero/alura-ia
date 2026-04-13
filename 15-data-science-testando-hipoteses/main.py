# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
# %%
experimento_lampadas_natalinas = pd.read_csv('./data/experimento_lampadas_natalinas.csv')
dados_alturas = pd.read_csv('./data/dados_alturas.csv')
dados_vida_lampada = pd.read_csv('./data/dados_vida_lampada.csv')
dados_idade_aposentadoria = pd.read_csv('./data/dados_idade_aposentadoria.csv')

# %%
plt.subplots(figsize=(20, 5))
plt.subplot(131)
plt.title('Distribuição de Idade de Aposentadoria')
plt.xlabel('Idade de Aposentadoria')
plt.ylabel('Frequência')
plt.hist(dados_idade_aposentadoria, bins=30, alpha=0.7, color='blue')

plt.subplot(132)
plt.hist(dados_vida_lampada, bins=30, alpha=0.7, color='purple')
plt.title('Tempo de Vida de uma Lâmpada')
plt.xlabel('Tempo de Vida (horas)')

plt.subplot(133)
plt.hist(dados_alturas, bins=30, alpha=0.7, color='green')
plt.title('Alturas dos Funcionários')
plt.xlabel('Altura (cm)')

plt.show()

# %%
def func_media(dados, coluna, n, qnt):
    return [dados[coluna].sample(n, replace = True).mean() for _ in range(qnt)]

# %%
media_idade = func_media(dados_idade_aposentadoria, 'idade', 100, 100000)
media_duracao = func_media(dados_vida_lampada, 'duracao', 100, 100000)
media_altura = func_media(dados_alturas, 'alturas', 100, 100000)


# %%
print(statistics.mean(media_idade))
print(statistics.mean(media_duracao))
print(statistics.mean(media_altura))
# %%
