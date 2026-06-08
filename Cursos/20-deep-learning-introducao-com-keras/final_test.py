# %% [markdown]
# # Testando o Modelo Carregado
# Este trecho simula o uso do modelo em produção, carregando o arquivo .keras salvo anteriormente.

# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. CARREGAR O MODELO E OS DADOS DE TESTE
# Carregamos o modelo exatamente como ele estava após o treino
modelo_carregado = keras.models.load_model('modelo_fashion_mnist.keras')

# Precisamos dos dados de teste e das classes para validar
(_, _), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ['Camiseta', 'Calça', 'Pullover', 'Vestido', 'Casaco', 
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

# IMPORTANTE: A normalização deve ser idêntica à do treino
test_images_norm = test_images / 255.0

# %%
# 2. RODAR PREDIÇÕES EM MASSA
# O modelo processa todas as imagens de teste de uma vez
predicoes = modelo_carregado.predict(test_images_norm)

# %%
# 3. VISUALIZAÇÃO DE RESULTADOS (O "TESTE REAL")
# Vamos plotar as primeiras 15 imagens e ver o que o modelo previu
plt.figure(figsize=(12, 8))

for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    
    # Identifica o índice da maior probabilidade
    indice_predito = np.argmax(predicoes[i])
    indice_real = test_labels[i]
    
    # Define a cor do texto: azul se acertou, vermelho se errou
    cor = 'blue' if indice_predito == indice_real else 'red'
    
    plt.xlabel(f"P: {class_names[indice_predito]}\n(R: {class_names[indice_real]})", color=cor)

plt.tight_layout()
plt.suptitle("Testes do Modelo Carregado (Azul = Acerto | Vermelho = Erro)", fontsize=16, y=1.05)
plt.show()

# %%
# 4. MÉTRICAS TÉCNICAS NO MODELO CARREGADO
perda, acuracia = modelo_carregado.evaluate(test_images_norm, test_labels, verbose=0)
print(f"Relatório do Modelo Carregado:")
print(f"- Acurácia: {acuracia * 100:.2f}%")
print(f"- Perda (Loss): {perda:.4f}")
# %%
