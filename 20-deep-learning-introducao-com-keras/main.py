# %% [markdown]
# # Classificação de Peças de Roupa com Deep Learning
# Projeto utilizando Fashion MNIST e Keras para estudo de redes neurais densas.

# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# %%
# 1. CARREGAMENTO DOS DADOS
print("Carregando o Fashion MNIST...")
dataset = keras.datasets.fashion_mnist
((train_images, train_labels), (test_images, test_labels)) = dataset.load_data()

# Definição das categorias para exibição amigável
class_names = ['Camiseta', 'Calça', 'Pullover', 'Vestido', 'Casaco', 
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

# %%
# 2. PRÉ-PROCESSAMENTO (NORMALIZAÇÃO)
# Essencial: Normalizar ambos os datasets para a escala [0, 1]
# Isso garante que os pesos da rede não explodam durante o treinamento e teste
train_images = train_images / 255.0
test_images = test_images / 255.0

# %%
# 3. CONSTRUÇÃO DO MODELO
# Utilizando a API Sequential com Input layer explícita (Padrão Keras 3)
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    
    # Camada de Flatten: transforma matriz 2D em vetor 1D
    keras.layers.Flatten(),
    
    # Camada Densa: 256 neurônios com ativação ReLU para capturar padrões não-lineares
    keras.layers.Dense(256, activation='relu'),
    
    # Dropout: Técnica de regularização para evitar overfitting
    keras.layers.Dropout(0.2),
    
    # Camada de Saída: 10 neurônios com Softmax para obter probabilidades por classe
    keras.layers.Dense(10, activation='softmax')
])

# Compilação com otimizador Adam (eficiente em termos de memória e processamento)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# 4. TREINAMENTO DO MODELO
# Reservando 20% dos dados de treino para validação em tempo real
history = model.fit(
    train_images, 
    train_labels, 
    epochs=10, 
    validation_split=0.2,
    verbose=1
)

# %%
# 5. AVALIAÇÃO DE DESEMPENHO
plt.figure(figsize=(10, 4))

# Plotando Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

# Plotando Perda (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# 6. TESTES E PREDIÇÕES INDIVIDUAIS
# Avaliando a performance final com dados de teste (completamente novos para o modelo)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f'\n[INFO] Acurácia Final no Teste: {test_acc*100:.2f}%')

# Realizando predições
predictions = model.predict(test_images)

# Exemplo: Comparando resultado do índice 1 (que é um Vestido)
index = 1
predicted_class = np.argmax(predictions[index])
real_class = test_labels[index]

print(f'\n--- Resultado do Teste (Índice {index}) ---')
print(f'Predição: {class_names[predicted_class]}')
print(f'Real: {class_names[real_class]}')

# %%
# 7. SALVAMENTO DO MODELO
# Salvando no formato nativo .keras para persistência
model.save('modelo_fashion_mnist.keras')
print("\nModelo salvo como 'modelo_fashion_mnist.keras'")
# %%
