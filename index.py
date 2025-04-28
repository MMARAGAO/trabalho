# %% [markdown]
"""
# Classificação de Pneumonia em Raios-X Torácicos

**Objetivo**: Desenvolver um modelo de Deep Learning para classificar imagens de raios-X torácicos em "Normal" ou "Pneumonia".

**Dataset**: Chest X-Ray Images (Pneumonia) do Kaggle, com a seguinte estrutura:
- `chest_xray/train/`
  - `NORMAL/`
  - `PNEUMONIA/`
- `chest_xray/test/`
  - `NORMAL/`
  - `PNEUMONIA/`
- `chest_xray/val/`
  - `NORMAL/`
  - `PNEUMONIA/`

**Base Teórica**: Inspirado no artigo CheXNet (Rajpurkar et al., 2017) que utilizou DenseNet-121 para alcançar desempenho de nível radiologista.
"""
# %%
# Importação de bibliotecas
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Verificando a estrutura dos diretórios
base_dir = 'C:/Users/MMARAGAO/Desktop/trabalho/chest_xray/chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

print("Diretório de treino:", os.listdir(train_dir))
print("Diretório de validação:", os.listdir(val_dir))
print("Diretório de teste:", os.listdir(test_dir))

# %%
# Análise exploratória - Contagem de imagens em cada conjunto
def count_images(directory):
    normal = len(os.listdir(os.path.join(directory, 'NORMAL')))
    pneumonia = len(os.listdir(os.path.join(directory, 'PNEUMONIA')))
    return normal, pneumonia

train_normal, train_pneumonia = count_images(train_dir)
val_normal, val_pneumonia = count_images(val_dir)
test_normal, test_pneumonia = count_images(test_dir)

print(f"Treino - Normal: {train_normal}, Pneumonia: {train_pneumonia}")
print(f"Validação - Normal: {val_normal}, Pneumonia: {val_pneumonia}")
print(f"Teste - Normal: {test_normal}, Pneumonia: {test_pneumonia}")

# Criando um gráfico de barras
labels = ['Treino', 'Validação', 'Teste']
normal_counts = [train_normal, val_normal, test_normal]
pneumonia_counts = [train_pneumonia, val_pneumonia, test_pneumonia]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, normal_counts, width, label='Normal')
rects2 = ax.bar(x + width/2, pneumonia_counts, width, label='Pneumonia')

ax.set_ylabel('Número de Imagens')
ax.set_title('Distribuição de Imagens por Classe')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

# %%
# Pré-processamento e aumento de dados
img_size = 224
batch_size = 32

# Data augmentation para o conjunto de treino
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Para validação e teste, apenas normalização
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Geradores de dados
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# %%
# Visualização de algumas imagens do dataset
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

sample_images, _ = next(train_generator)
plot_images(sample_images[:5])

# %%
# Construção do modelo baseado em DenseNet121
def build_model():
    # Carregar DenseNet121 pré-treinada no ImageNet
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    
    # Congelar as camadas da base
    base_model.trainable = False
    
    # Construir o modelo completo
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compilar o modelo
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

model = build_model()
model.summary()

# %%
# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# %%
# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# %%
# Visualização do histórico de treinamento
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    # Gráfico de loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# %%
# Avaliação no conjunto de teste
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print(f'\nAcurácia no teste: {test_acc:.2%}')
print(f'AUC no teste: {test_auc:.2%}')

# %%
# Matriz de confusão e relatório de classificação
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Previsões no conjunto de teste
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred = (y_pred > 0.5).astype(int)
y_true = test_generator.classes

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pneumonia'], 
            yticklabels=['Normal', 'Pneumonia'])
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.title('Matriz de Confusão')
plt.show()

# Relatório de classificação
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# %%
# Grad-CAM para interpretabilidade (opcional)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Criar modelo que mapeia a imagem de entrada para as ativações
    # da última camada convolucional e as saídas preditas
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Calcular o gradiente da classe predita em relação à
    # última camada convolucional
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradiente da saída em relação à última camada convolucional
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vetor onde cada entrada é a intensidade média do gradiente
    # sobre um canal específico
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplicar cada canal no array da camada convolucional
    # pelo "quão importante" esse canal é para a classe predita
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizar o heatmap entre 0 e 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Exemplo de Grad-CAM em uma imagem
def display_gradcam(img_path, heatmap, alpha=0.4):
    # Carregar a imagem original
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Redimensionar o heatmap para o mesmo tamanho da imagem
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)

    # Superpor o heatmap na imagem original
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Mostrar a imagem resultante
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

# Pegar uma imagem de exemplo
sample_img_path = os.path.join(test_dir, 'NORMAL/IM-0001-0001.jpeg')
img_array = tf.keras.preprocessing.image.load_img(
    sample_img_path, target_size=(img_size, img_size))
img_array = tf.keras.preprocessing.image.img_to_array(img_array)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Gerar heatmap
last_conv_layer_name = 'conv5_block16_concat'
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Mostrar resultado
display_gradcam(sample_img_path, heatmap)