import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import numpy as np

# Caminho da pasta das imagens
pasta_imagens = 'C:/Users/Usuario/Pictures/coderace/Multi-Class Driver Behavior Image Dataset/com_sono'

# Configurar o gerador de aumentação
datagen = ImageDataGenerator(
    rotation_range=19,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.6, 1.5],
    shear_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Quantidade de novas imagens geradas de cada imagem das pastas
quantidade_aumentos = 5

# Processar todas as imagens da pasta
for nome_arquivo in os.listdir(pasta_imagens):
    caminho_imagem = os.path.join(pasta_imagens, nome_arquivo)
    
    # Verifica se é uma imagem válida
    if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Carrega a imagem
            img = load_img(caminho_imagem)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Gera imagens aumentadas
            contador = 0
            for batch in datagen.flow(img_array, batch_size=1):
                nova_imagem = array_to_img(batch[0])
                novo_nome = f"nova_{contador}_{nome_arquivo}"
                nova_imagem.save(os.path.join(pasta_imagens, novo_nome))
                contador += 1
                if contador >= quantidade_aumentos:
                    break

        except Exception as e:
            print(f"Erro ao processar {nome_arquivo}: {e}")
