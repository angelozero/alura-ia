# %%
# Importando a lib do pandas
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Recuperando dados do csv
df = pd.read_csv("./data/imigrantes_canada.csv")

# %%
# Exibindo dados da planilha
df.head()

# %%
# Analisando as tendencias de imigração
df.set_index("País", inplace=True)
anos = list(map(str, range(1980, 2014)))
brasil = df.loc['Brasil', anos]
brasil_dict = {'ano': brasil.index.tolist(), 'imigrantes': brasil.values.tolist()}
dados_brasil = pd.DataFrame(brasil_dict)


# %%
# Criando um grafico com PyPlot
plt.figure(figsize=(7.5, 3))
plt.title('Imigração Brasil <> Canadá')
plt.xlabel('Ano')
plt.ylabel('Imigrantes')
plt.plot(dados_brasil['ano'], dados_brasil['imigrantes'])
plt.xticks(dados_brasil['ano'][::5])
plt.show()



# %%
fig, ax = plt.subplots(figsize=(7.5, 3))
ax.plot(dados_brasil['ano'], dados_brasil['imigrantes'])
ax.set_title('Imigração Brasil <-> Canadá')
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
plt.show()
# %%
