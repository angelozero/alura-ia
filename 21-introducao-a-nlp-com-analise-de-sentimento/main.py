# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# %%
resenha = pd.read_csv("./data/imdb-reviews-pt-br.csv")
resenha.head()

# %%
treino_t, teste_t, classe_treino_t, classe_teste_t = train_test_split(
    resenha["text_pt"], resenha["sentiment"], random_state=42
)

print(f"Treino: {len(treino_t)}")
print(f"Teste: {len(teste_t)}")


# %%
print(resenha["sentiment"].value_counts())

classificacao = resenha["sentiment"].replace({"neg": 0, "pos": 1})
resenha["classificacao"] = classificacao
resenha.head()

# %%
# Aplicando LPN (NLP) Linguagem Natural Processamento - Análise de Sentimentos
# Bag of Words (BoW) - Sacola de Palavras
# | Frase                     | O | filme | é | muito | bom | ruim |

# | O filme é muito bom       | 1 | 1     | 1 | 1     | 1   | 0    |
# | O filme é muito ruim      | 1 | 1     | 1 | 1     | 0   | 1    |
# | O filme é muito muito bom | 1 | 1     | 1 | 2     | 1   | 0    |
# | O filme é péssimo         | 1 | 1     | 1 | 0     | 0   | 0    |

texto = [
    "O filme é muito bom",
    "O filme é muito ruim",
    "O filme é muito muito bom",
    "O filme é péssimo",
]
vetorizador_test = CountVectorizer(lowercase=False)
bag_of_words_test = vetorizador_test.fit_transform(texto)
print(vetorizador_test.get_feature_names_out())
print(bag_of_words_test.toarray())
matriz = pd.DataFrame(
    bag_of_words_test.toarray(), columns=vetorizador_test.get_feature_names_out()
)
print(matriz)


# %%
def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizador = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizador.fit_transform(texto[coluna_texto])
    print(bag_of_words.shape)

    treino, teste, classe_treino, classe_teste = train_test_split(
        bag_of_words, texto[coluna_classificacao], random_state=42
    )
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    acuracia = regressao_logistica.score(teste, classe_teste)
    print(f"Acurácia: {acuracia * 100:.2f}%")
    return acuracia


# %%
print(classificar_texto(resenha, "text_pt", "classificacao"))

# %%
todas_as_palavras = " ".join([texto for texto in resenha["text_pt"]])
nuvem_de_palavras = WordCloud(width=800, height=400, collocations=False).generate(todas_as_palavras)

plt.figure(figsize=(15, 7.5))
plt.imshow(nuvem_de_palavras, interpolation="bilinear")
plt.axis("off")
plt.show()


# %%
