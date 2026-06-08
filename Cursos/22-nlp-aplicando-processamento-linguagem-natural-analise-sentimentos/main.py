# %%
# 1. Instalação e importação das bibliotecas necessárias
import sys
!{sys.executable} -m pip install seaborn wordcloud scikit-learn nltk unidecode

import pandas as pd
import seaborn as sns
import unidecode
import matplotlib.pyplot as plt
import joblib
import nltk
from nltk import ngrams

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
from nltk import tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('all')

# %%
# 2. Carregamento e exploração inicial dos dados
df = pd.read_csv('./data/dataset_avaliacoes.csv')

print("Número total de registros e colunas:", df.shape)

positive_count = df[df['sentimento'] == 'positivo'].shape[0]
negative_count = df[df['sentimento'] == 'negativo'].shape[0]

print("Número de avaliações positivas:", positive_count)
print("Número de avaliações negativas:", negative_count)
print("Exemplo de avaliação positiva:", df.avaliacao[0])
print("Exemplo de avaliação negativa:", df.avaliacao[1])

df.head()

# %%
# 3. Bag of Words (BoW) — representação numérica do texto usando contagem das 50 palavras mais frequentes
vectorizer = CountVectorizer(max_features=50)
bag_of_words = vectorizer.fit_transform(df['avaliacao'])

# %%
# 4. Primeiro modelo — Regressão Logística com texto bruto (sem tratamento)
# Regressão Logística: algoritmo de classificação binária baseado na função sigmoid,
# que mapeia qualquer valor real para [0, 1], representando a probabilidade de uma classe.
X_train, X_test, y_train, y_test = train_test_split(
    bag_of_words,
    df['sentimento'],
    random_state=4978
)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

accuracy = logistic_regression.score(X_test, y_test)
print(f"Acurácia do modelo (texto bruto): {accuracy * 100:.2f}%")

# %%
# 5. Nuvem de palavras geral — visualização das palavras mais frequentes (positivas + negativas)
all_words = ' '.join(df['avaliacao'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_font_size=75,
    collocations=False
).generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
# 6. Função reutilizável para gerar nuvem de palavras filtrada por sentimento
def generate_word_cloud(dataframe, text_column, sentiment):
    """
    Gera e exibe uma nuvem de palavras para um sentimento específico.

    Parâmetros:
        dataframe    — DataFrame com os dados
        text_column  — nome da coluna que contém o texto
        sentiment    — valor do sentimento a filtrar ('positivo' ou 'negativo')
    """
    filtered_texts = dataframe.query(f"sentimento == '{sentiment}'")[text_column]
    combined_text = ' '.join(filtered_texts)

    cloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_font_size=75,
        collocations=False
    ).generate(combined_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Palavras mais frequentes em avaliações {sentiment}')
    plt.show()

# %%
# 7. Nuvem de palavras por sentimento — negativo e positivo
generate_word_cloud(df, 'avaliacao', 'negativo')
generate_word_cloud(df, 'avaliacao', 'positivo')

# %%
# 8. Tokenização e análise de frequência de palavras
# Tokenização: dividir o texto em unidades menores (tokens/palavras) para análise no NLP.
whitespace_tokenizer = tokenize.WhitespaceTokenizer()
all_tokens = whitespace_tokenizer.tokenize(all_words)

# %%
# 9. Função para calcular e exibir a frequência das palavras mais comuns em gráfico de barras
def plot_word_frequency(tokens, top_n=10):
    """
    Calcula a frequência dos tokens e exibe um gráfico de barras.

    Parâmetros:
        tokens — lista de tokens (palavras)
        top_n  — quantidade de palavras mais frequentes a exibir
    """
    frequency = FreqDist(tokens)
    print(frequency.most_common(top_n))

    df_frequency = pd.DataFrame(
        frequency.most_common(top_n),
        columns=['Palavra', 'Frequência']
    )
    print(df_frequency)

    plt.figure(figsize=(20, 5))
    sns.barplot(x='Palavra', y='Frequência', data=df_frequency)
    plt.title(f'Frequência das {top_n} palavras mais comuns')
    plt.xlabel('Palavra')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.show()

# %%
# 10. Gráfico de frequência com o texto bruto (sem nenhum tratamento)
plot_word_frequency(all_tokens)

# %%
# 11. Tratamento 1 — remoção de stop words (palavras comuns que não agregam significado, ex: "de", "a", "o")
stop_words_pt = nltk.corpus.stopwords.words('portuguese')

processed_step_1 = []
for review in df['avaliacao']:
    tokens = whitespace_tokenizer.tokenize(review)
    filtered = [word for word in tokens if word.lower() not in stop_words_pt]
    processed_step_1.append(' '.join(filtered))

df['tratamento_1'] = processed_step_1
df.head()

# %%
# 12. Função reutilizável para classificar texto com Regressão Logística e exibir acurácia
def classify_text(dataframe, text_column, label_column):
    """
    Treina um modelo de Regressão Logística e exibe a acurácia.

    Parâmetros:
        dataframe    — DataFrame com os dados
        text_column  — coluna com o texto já tratado
        label_column — coluna com os rótulos de sentimento
    """
    vector = CountVectorizer(lowercase=False, max_features=50)
    bow = vector.fit_transform(dataframe[text_column])

    X_train, X_test, y_train, y_test = train_test_split(
        bow,
        dataframe[label_column],
        random_state=4978
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do modelo ({text_column}): {accuracy * 100:.2f}%")

# %%
# 13. Tratamento 2 — remoção de pontuação e caracteres não-alfabéticos
processed_step_2 = []
for text in df['tratamento_1']:
    tokens = whitespace_tokenizer.tokenize(text)
    cleaned = [w for w in tokens if w.isalpha() and w.lower() not in stop_words_pt]
    processed_step_2.append(' '.join(cleaned))

df['tratamento_2'] = processed_step_2
df.head()

# %%
# 14. Tratamento 3 — remoção de acentos (texto e stop words normalizados com unidecode)
without_accents = [unidecode.unidecode(text) for text in df['tratamento_2']]
stop_words_no_accents = [unidecode.unidecode(word) for word in stop_words_pt]

df['tratamento_3'] = without_accents

processed_step_3 = []
for text in df['tratamento_3']:
    tokens = whitespace_tokenizer.tokenize(text)
    cleaned = [w for w in tokens if w.isalpha() and w.lower() not in stop_words_no_accents]
    processed_step_3.append(' '.join(cleaned))

df['tratamento_3'] = processed_step_3
df.head()

# %%
# 15. Gráfico de frequência após remoção de acentos
plot_word_frequency(whitespace_tokenizer.tokenize(' '.join(df['tratamento_3'])))

# %%
# 16. Tratamento 4 — conversão para letras minúsculas (normalização final)
processed_step_4 = []
for text in df['tratamento_3']:
    tokens = whitespace_tokenizer.tokenize(text)
    cleaned = [w.lower() for w in tokens if w.isalpha() and w.lower() not in stop_words_no_accents]
    processed_step_4.append(' '.join(cleaned))

df['tratamento_4'] = processed_step_4
df.head()

# %%
# 17. Gráfico de frequência após normalização completa (minúsculas)
plot_word_frequency(whitespace_tokenizer.tokenize(' '.join(df['tratamento_4'])))

# %%
# 18. Avaliação de acurácia do modelo com o texto totalmente tratado
# Nota: a acurácia pode ser menor que a do texto bruto (etapa 4) porque aqui
# usamos lowercase=False no CountVectorizer (o texto já foi normalizado manualmente)
# e o vocabulário de 50 features muda conforme o pré-processamento aplicado.
# Isso é esperado — o objetivo é comparar os resultados entre as abordagens.
classify_text(df, 'tratamento_4', 'sentimento')

# %%
# 19. Reduzindo a palavra ao radical dela (stemming)
stemmer = nltk.RSLPStemmer()
processed_step_5 = []
for text in df['tratamento_4']:
    tokens = whitespace_tokenizer.tokenize(text)
    stemmed = [stemmer.stem(w) for w in tokens if w.isalpha() and w.lower() not in stop_words_no_accents]
    processed_step_5.append(' '.join(stemmed))  

df['tratamento_5'] = processed_step_5
df.head()

# %%
# 20. Avaliação de acurácia do modelo com o texto tratado e com stemming
classify_text(df, 'tratamento_5', 'sentimento')

# %%
# 21. Dando peso para as palavras sejam negativas e/ou positivas com a técnica TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(lowercase=False, max_features=50)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['avaliacao'])
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix,
    df['sentimento'],
    random_state=4978
)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Acurácia do modelo (TF-IDF): {accuracy * 100:.2f}%")

# %%
# 21. Dando peso para as palavras sejam negativas e/ou positivas com a técnica TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_final_vectorizer = TfidfVectorizer(lowercase=False, max_features=50)
tfidf_final_matrix = tfidf_final_vectorizer.fit_transform(df['tratamento_5'])
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_final_matrix,
    df['sentimento'],
    random_state=4978
)
model_final = LogisticRegression()
model_final.fit(X_train, y_train)
accuracy_final = model_final.score(X_test, y_test)
print(f"Acurácia do modelo (TF-IDF): {accuracy_final * 100:.2f}%")


# %%
# 22. Capturando contextos e expressões idiomáticas com n-grams (combinações de palavras)
tfidf_features = TfidfVectorizer(lowercase=False, max_features=1000, ngram_range=(1, 2))
tfidf_features_vector = tfidf_features.fit_transform(df['tratamento_5'])
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_features_vector,
    df['sentimento'],
    random_state=4978
)
model_ngrams = LogisticRegression()
model_ngrams.fit(X_train, y_train)
accuracy_ngrams = model_ngrams.score(X_test, y_test)
print(f"Acurácia do modelo (TF-IDF + n-grams): {accuracy_ngrams * 100:.2f}%")

# %%
# 23. Obtendo os pesos dos sentimentos negativos e positvos para as palavras mais importantes do modelo
pesos = pd.DataFrame(
    model_ngrams.coef_[0].T, 
    index=tfidf_features.get_feature_names_out(), 
    columns=['Peso']
).sort_values(by='Peso', ascending=False)

# %%
# 23.1 Positivo
print(pesos.head(50))

# %%
# 23.2 Negativo
print(pesos.tail(50))

# %%
# 24. Salvando o modelo e vetorizador para uso futuro
joblib.dump(model_ngrams, './pkl/modelo_sentimentos.pkl')
joblib.dump(tfidf_features, './pkl/vetorizador_sentimentos.pkl')


# %%
# 25. Carregando os modelos salvo para classificar uma nova avaliação
vectorizer_generated = joblib.load('./pkl/vetorizador_sentimentos.pkl')
model_generated = joblib.load('./pkl/modelo_sentimentos.pkl')

# %%
# 26. Criando uma função para processar todas as etapas de pré-processamento e classificação de uma nova avaliação
irrelevant_words = nltk.corpus.stopwords.words('portuguese')
token_pontuation = tokenize.WordPunctTokenizer
steamer = nltk.RSLPStemmer()
def preprocess_and_classify(avaliation):
    """
    Processa o texto de entrada e classifica o sentimento usando o modelo treinado.

    Parâmetros:
        avaliation — string com a avaliação a ser classificada

    Retorna:
        string com a classificação do sentimento ('positivo' ou 'negativo')
    """
    # Pré-processamento
    # Passo 1
    tokens = token_pontuation().tokenize(avaliation)
    
    # Passo 2 - Removendo stop words
    processed_phrase = [word for word in tokens if word.lower() not in irrelevant_words]
    
    # Passo 3 - Removendo pontuação e caracteres não-alfabéticos
    processed_phrase = [w for w in processed_phrase if w.isalpha()]
    
    # Passo 4 - Removendo acentuacão
    processed_phrase = [unidecode.unidecode(w) for w in processed_phrase]
    
    # Passo 5 - Recuperando o radical da palavra (stemming)
    processed_phrase = [steamer.stem(w) for w in processed_phrase]
    
    return ' '.join(processed_phrase)
    
# %%
# 27. Testando o modelo com uma nova avaliação
# Novas avaliações para prever
new_evaluations = ["Ótimo produto, super recomendo!",
                    "A entrega atrasou muito! Estou decepcionado com a compra",
                    "Muito satisfeito com a compra. Além de ter atendido as expectativas, o preço foi ótimo",
                    "Horrível!!! O produto chegou danificado e agora estou tentando fazer a devolução.",
                    '''Rastreando o pacote, achei que não fosse recebê-lo, pois, na data prevista, estava sendo entregue em outra cidade.
                    Mas, no fim, deu tudo certo e recebi o produto. Produto de ótima qualidade, atendendo bem as minhas necessidades e por um preço super em conta. Recomendo.''']

new_evaluations_processed = [preprocess_and_classify(evaluation) for evaluation in new_evaluations]
new_evaluations_vectorized = vectorizer_generated.transform(new_evaluations_processed)
predictions = model_generated.predict(new_evaluations_vectorized)

df_predictions = pd.DataFrame({
    'Avaliação ': new_evaluations,
    ' Sentimento': predictions
})
print(df_predictions)
# %%
