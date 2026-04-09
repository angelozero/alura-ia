# %%
# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %% 
# Carregando o arquivo CSV com os dados de avaliações
df = pd.read_csv('./data/dataset_avaliacoes.csv')

df.head()

# %%
# Verificando o número total de registro com a quantidade de colunas
print("O numero total de registros com colunas é: ", df.shape)

# %%
# Exibindo avaliações positivas e negativas
print('Número de avaliações positivas:', df[df['sentimento'] == 'positivo'].shape[0])
print('Número de avaliações negativas:', df[df['sentimento'] == 'negativo'].shape[0])
print('Avaliacao positiva: ', df.avaliacao[0])
print('Avaliacao negativa: ', df.avaliacao[1])

# %%
# Criando um Bag of Words (BoW) para representar as avaliações
vectorizer = CountVectorizer(max_features=50)
bag_of_words = vectorizer.fit_transform(df['avaliacao'])

# %%
# Exibindo as palavras únicas e suas contagens
matrix = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vectorizer.get_feature_names_out())
print(matrix)

# %%
# Classificando se o sentimento é positivo ou negativo com base na contagem de palavras
## Separando os conjuntos em treinamento e teste
### X_train: Conjunto de dados de treinamento (features)
### X_test: Conjunto de dados de teste (features)
### y_train: Conjunto de dados de treinamento (rótulos)
### y_test: Conjunto de dados de teste (rótulos)
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, df['sentimento'], random_state=4212)

# %%
# Aplicando regressão logística para classificar os sentimentos
### Algoritmo de probabilidade que é usado para classificação binária para identificar se uma classe pertence a um grupo ou não. Ele é baseado na função logística, que mapeia qualquer valor real para um valor entre 0 e 1, representando a probabilidade de uma classe ser positiva ou negativa.
logistic_regression = LogisticRegression()

### Aplicando treinamento do modelo de regressão logística usando os dados de treinamento (X_train e y_train). O modelo aprenderá a associar as características (contagem de palavras) com os rótulos de sentimento (positivo ou negativo).
logistic_regression.fit(X_train, y_train)

#%%
# Avaliando o modelo usando os dados de teste (X_test e y_test) para obter uma métrica de acertos. Acuracy é a proporção de previsões corretas em relação ao total de previsões feitas. Ela é calculada como o número de previsões corretas dividido pelo número total de previsões.
accuracy = logistic_regression.score(X_test, y_test)
print("Acurácia do modelo: ", accuracy)
