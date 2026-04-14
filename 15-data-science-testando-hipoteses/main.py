# %%
import sys
!{sys.executable} -m pip install statsmodels

# %%
# Extraindo conclusões válidas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy import stats
from statsmodels.stats.weightstats import ztest

lampadas_natalinas = pd.read_csv("./data/experimento_lampadas_natalinas.csv")
dados_alturas = pd.read_csv("./data/dados_alturas.csv")
dados_vida_lampada = pd.read_csv("./data/dados_vida_lampada.csv")
dados_idade_aposentadoria = pd.read_csv("./data/dados_idade_aposentadoria.csv")

plt.subplots(figsize=(20, 5))
plt.subplot(131)
plt.title("Distribuição de Idade de Aposentadoria")
plt.xlabel("Idade de Aposentadoria")
plt.ylabel("Frequência")
plt.hist(dados_idade_aposentadoria, bins=30, alpha=0.7, color="blue")

plt.subplot(132)
plt.hist(dados_vida_lampada, bins=30, alpha=0.7, color="purple")
plt.title("Tempo de Vida de uma Lâmpada")
plt.xlabel("Tempo de Vida (horas)")

plt.subplot(133)
plt.hist(dados_alturas, bins=30, alpha=0.7, color="green")
plt.title("Alturas dos Funcionários")
plt.xlabel("Altura (cm)")

plt.show()


def func_media(dados, coluna, n, qnt):
    # Converte a coluna para um array NumPy para máxima velocidade
    valores = dados[coluna].values

    # Gera uma matriz de índices aleatórios (qnt linhas, n colunas) de uma só vez
    indices = np.random.randint(0, len(valores), size=(qnt, n))

    # Mapeia os índices para os valores e calcula a média ao longo das colunas (axis=1)
    return np.mean(valores[indices], axis=1)


media_idade = func_media(dados_idade_aposentadoria, "idade", 100, 100000)
media_duracao = func_media(dados_vida_lampada, "duracao", 100, 100000)
media_altura = func_media(dados_alturas, "alturas", 100, 100000)

print("Medias originais")
print(dados_idade_aposentadoria["idade"].mean())
print(dados_vida_lampada["duracao"].mean())
print(dados_alturas["alturas"].mean())

print("")

print("Medias Amostrais")
print(statistics.mean(media_idade))
print(statistics.mean(media_duracao))
print(statistics.mean(media_altura))

plt.subplots(figsize=(20, 5))
plt.subplot(131)
plt.title("Distribuição de Idade de Aposentadoria")
plt.xlabel("Idade de Aposentadoria")
plt.ylabel("Frequência")
plt.hist(media_idade, bins=30, alpha=0.7, color="blue")

plt.subplot(132)
plt.hist(media_duracao, bins=30, alpha=0.7, color="purple")
plt.title("Tempo de Vida de uma Lâmpada")
plt.xlabel("Tempo de Vida (horas)")

plt.subplot(133)
plt.hist(media_altura, bins=30, alpha=0.7, color="green")
plt.title("Alturas dos Funcionários")
plt.xlabel("Altura (cm)")

plt.show()

# %%
# Intervalo de confiança - lâmpada - Duração de 1200 para 1732 ?
# Calculando o erro padrão
# Descobrindo quantas amostras estão dentro do intervalo de erros padroes

media_antiga_amostra = media_duracao.mean()  # 1732
media_nova_amostra = 1200

plt.hist(media_duracao, bins=30, alpha=0.7, color="purple")
plt.title("Duração média de uma Lâmpada")
plt.xlabel("Tempo de Vida médio(horas)")
plt.annotate(
    ".",
    xy=(media_nova_amostra, 0),
    xytext=(media_nova_amostra, 100),
    fontsize=8,
    arrowprops=dict(facecolor="green"),
)
plt.show()

duracao_amostras = pd.DataFrame({"media_duracao": media_duracao})

media_das_medias = duracao_amostras["media_duracao"].mean()
erro_padrao = duracao_amostras["media_duracao"].std()

print("Media       : ", media_das_medias)
print("Erro Padrao : ", erro_padrao)

plt.hist(media_duracao, bins=30, alpha=0.7, color="Blue")
plt.title("Duração média de uma Lâmpada")
plt.xlabel("Tempo de Vida médio(horas)")
plt.axvline(media_das_medias, color="Yellow")
plt.axvline(media_das_medias + (3 * erro_padrao), color="Red")
plt.axvline(media_das_medias - +(3 * erro_padrao), color="Red")
plt.annotate(
    ".",
    xy=(media_nova_amostra, 0),
    xytext=(media_nova_amostra, 100),
    fontsize=8,
    arrowprops=dict(facecolor="green"),
)
plt.show()


quantidade_observacoes_intervalo = duracao_amostras[
    (duracao_amostras > media_das_medias - (3 * erro_padrao))
    & (duracao_amostras < media_das_medias + (3 * erro_padrao))
]

resultado = (quantidade_observacoes_intervalo.count() / duracao_amostras.count()) * 100

print("Media: ", resultado["media_duracao"])

# %%
# Testando suposições - Media de duração de uma lâmpada natalina é de 1570 horas
# Calculando se dentro do intervalo rejeitamos ou aceitamos a hipotese de 1570 horas
media_hipotese_base = 1570
lampadas_natalinas["duracao"].mean()


confianca = 0.95  # valor dado pelo cientista de dado
desvio_padrao_populacional = 105 # processo da fabricacao previso
tamanho_amostra = len(lampadas_natalinas["duracao"])

intervalo = stats.norm.interval(
    confianca, loc=1570, scale=(desvio_padrao_populacional / np.sqrt(tamanho_amostra))
)

print("Intervalo de confiança: 95% --- ", intervalo[0] , intervalo[1] )

# %%
# Erros associados a hipotese
# 05 % Nivel de confiança
# Teste Z
estatistica_do_teste, p_valor = ztest(x1 = lampadas_natalinas['duracao'], value = 1570, alternative = "two-sided")

print("P Valor: ", p_valor)

print("Conclusão:", "Rejeitar a hipótese nula" if p_valor < 0.05 else "Não rejeita a hipótese nula")

# %%
