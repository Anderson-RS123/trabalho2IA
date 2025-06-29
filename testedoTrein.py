from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Caminho da imagem para teste
img_path = 'C:/Users/Usuario/Downloads/celular2.jpeg'  

# Caminho do modelo treinado
model_path = 'modelo_motorista_new.h5'

# Tamanho da imagem
img_width, img_height = 224, 224

model = load_model(model_path)

# Indicar as classes
class_labels = {
    0: 'com_sono',
    1: 'dirigindo_seguro',
    2: 'usando_celular'
}

img = image.load_img(img_path, target_size=(img_width, img_height))   # Carregar a imagem e redimensionar
img_array = image.img_to_array(img)  # Converter para array
img_array = img_array / 255.0   # Normalizar os pixels

# Expandir dimensão para simular um batch de tamanho 1
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)  # Predição
confidence = np.max(pred)  # Obter a maior probabilidade
class_idx = np.argmax(pred)   # Obter o índice da classe predita

categoria = class_labels[class_idx]
print(f"Resultado: {categoria} (confiança: {confidence*100:.2f}%)")