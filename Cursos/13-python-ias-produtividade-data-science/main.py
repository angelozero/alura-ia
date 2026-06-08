# %%
import pandas as pd

url_resumo_mundial = 'https://raw.githubusercontent.com/alura-cursos/IA-produtividade-DS/main/Dados/Resumo_mundial.csv'
url_producao = 'https://github.com/alura-cursos/IA-produtividade-DS/raw/main/Dados/Producao_total.csv'

# Importar o dataset de resumo mundial
resumo_mundial = pd.read_csv(url_resumo_mundial)

# Importar o dataset de produção
producao = pd.read_csv(url_producao)


# %%
resumo_mundial.corr()
# %%
# Esse curso usa chat gpt pra montar grafico, chato!