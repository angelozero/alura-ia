# Avaliando peças de roupa com Deep Learning e Keras
# %%
# Instalando dependencia do TensorFlow e Keras
import sys
!{sys.executable} -m pip install tensorflow

# %%
# Importando bibliotecas
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


 # %%
 # Carregando o dataset de roupas
dataset = keras.datasets.fashion_mnist

# %%
# O dataset já vem dividido em treino e teste, e as imagens são 28x28 pixels
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_labels.min()
train_labels.max()
total_de_classificacoes = 10
nomes_de_classificacoes = ['Camiseta', 'Calça', 'Pullover', 
                           'Vestido', 'Casaco', 'Sandália', 'Camisa', 
                           'Tênis', 'Bolsa', 'Bota']

for imagem in range(10):
    plt.subplot(2, 5, imagem+1)
    plt.imshow(train_images[imagem])
    plt.title(nomes_de_classificacoes[train_labels[imagem]])

# Normalizando os valores dos pixels para o intervalo de 0 a 1, para melhorar a performance do modelo
train_images = train_images / 255.0

# %%
# Criando o modelo de rede neural com 3 camadas: entrada, processamento e saída
# Peça de entrada === Camiseta

# Categoria   Percentual
# --------- | ----------
# Camiseta    67 %
# --------- | ----------
# Vestido     12 %
# --------- | ----------
# Saia        11 %
# --------- | ----------
# Calça       8 %
# --------- | ----------
# Bota        2 %

# A soma total de todas as categorias é 100 % e a categoria com maior percentual 
# é a camiseta, então o modelo irá classificar a imagem como camiseta
modelo = keras.Sequential([
    # Camada de entrada - Achatando as imagens de 28x28 pixels em um vetor de 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # Camada de processamento - Camada densa com 256 neurônios para avaliar todas as linhas de pixels
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    
    # Camada de saída - Camada densa com 10 neurônios para dizer qual a probabilidade da imagem pertencer a uma categoria especifica
    keras.layers.Dense(10, activation='softmax')
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modelo.fit(train_images, train_labels, epochs=10)

# %%
