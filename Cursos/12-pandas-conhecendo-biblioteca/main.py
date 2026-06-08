# %%
import pandas as pd

# %%
# Importando o csv
url = 'https://raw.githubusercontent.com/alura-cursos/pandas-conhecendo-a-biblioteca/main/base-de-dados/aluguel.csv'
dados = pd.read_csv(url, sep=';')

# %%
# Exibindo a tabela do csv
dados.head()

# %%
# Exibindo os tipos de dados
type(dados)

# %%
# Media de valor da coluna 'Valor'
dados['Valor'].mean()

# %%
# Agrupando 'Valor' pela coluna 'Tipo' e formando o valor
dados_formatados = dados.groupby('Tipo')[['Valor']].mean(numeric_only=True).sort_values('Valor').style.format({'Valor': 'R$ {:,.2f}'})

print(dados_formatados)

df_preco_tipo = dados.groupby('Tipo')[['Valor']].mean(numeric_only=True).sort_values('Valor')

# %%
# Exibindo grafico
df_preco_tipo.plot(kind='barh', figsize=(14, 10), color='purple')

# %%
# Recuperando imoveis
imoveis = dados['Tipo'].unique()

# %%
imoveis_comerciais = ['Conjunto Comercial/Sala', 
                      'Prédio Inteiro', 'Loja/Salão',
                      'Galpão/Depósito/Armazém', 
                      'Casa Comercial', 
                      'Terreno Padrão', 
                      'Box/Garagem',
                      'Loja Shopping/ Ct Comercial', 
                      'Chácara', 
                      'Loteamento/Condomínio',
                      'Sítio', 
                      'Pousada/Chalé', 
                      'Hotel', 
                      'Indústria']

# %%
# Recuperando apenas imoveis residencias
imoveis_residenciais = dados_filtrados = dados.query('@imoveis_comerciais not in Tipo')
imoveis_residenciais

imoveis_residenciais['Tipo'].unique()

# %%
# Exibindo grafico com apenas imoveis residenciais
imoveis_residenciais.groupby('Tipo')[['Valor']].mean(numeric_only=True).sort_values('Valor').plot(kind='barh', figsize=(14, 10), color='purple')


# %%
# Calulando o percentual de cada tipo de imovel na base 
imoveis_residenciais_percentual= imoveis_residenciais['Tipo'].value_counts(normalize=True).to_frame()

# %%
imoveis_residenciais_percentual.plot(kind='bar', figsize=(14, 10), color='green', xlabel='Tipo de Imóvel', ylabel='Percentual', title='Percentual de cada tipo de imóvel residencial')

# %%
# Trabalhando apenas com Apartamentos
apartamentos = imoveis_residenciais.query('Tipo == "Apartamento"')

# %%
# Lidando com dados nulos
apartamentos.isnull().sum()

# %%
apartamentos_nao_nulos = apartamentos.fillna(0)
apartamentos_nao_nulos.isnull().sum()


# %%
# Removendo linhas com valor == 0
registros_a_remover = apartamentos_nao_nulos.query('Valor == 0 | Condominio == 0').index
apartamentos.drop(apartamentos_nao_nulos, axis=0, inplace=True)

# %%
