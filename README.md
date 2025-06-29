# 🚗 Reconhecimento de Comportamento de Motoristas com IA

Este projeto utiliza redes neurais convolucionais (CNN) com TensorFlow para reconhecer comportamentos de motoristas por meio de imagens capturadas por câmeras em cabines de veículos. O objetivo é identificar situações como **uso de celular**, **sonolência** ou **condução segura**, contribuindo para sistemas de monitoramento mais inteligentes e preventivos.

## 📌 Motivação

A ideia surgiu a partir de uma necessidade real identificada no ambiente de trabalho do autor: monitorar motoristas em tempo real para evitar acidentes, melhorar a segurança e prevenir comportamentos de risco nas estradas.

## 🧠 Tecnologias e Bibliotecas

- **TensorFlow** – Para construção e treinamento do modelo CNN.
- **Keras** – Para gerenciamento das camadas e pipelines de dados.
- **NumPy** – Para manipulação de arrays e imagens.
- **Matplotlib** – Para visualização gráfica dos resultados.
- **ImageDataGenerator** – Para aumentação dos dados.

## 📁 Estrutura do Projeto

- `trabalho.py` – Treinamento do modelo CNN com TensorFlow.
- `aumentandoDados.py` – Aumentação das imagens para enriquecer o dataset.
- `testedoTrein.py` – Teste de uma imagem externa com o modelo treinado.
- `modelo_motorista.h5` – Arquivo gerado com os pesos do modelo (não incluído por padrão).

## 🖼️ Dataset

- Imagens coletadas manualmente da internet e organizadas em 3 categorias:
  - `dirigindo_seguro`
  - `com_sono`
  - `usando_celular`
- Após coleta, o dataset passou por aumentação com transformações como:
  - Rotação, deslocamento, brilho, zoom, espelhamento, etc.
- Foram utilizadas cerca de **50.000 imagens** após aumentação.

## 🔧 Parâmetros do Modelo

| Parâmetro      | Valor           |
|----------------|------------------|
| Tamanho da imagem (`img_size`) | 224x224 pixels |
| Lote de treinamento (`batch_size`) | 15 |
| Épocas (`epochs`) | 10 |
| Taxa de aprendizado (`lr`) | 0.0001 (ajustável) |
| Camadas CNN | 3 camadas convolucionais + densas |
| Saída | 3 classes (softmax) |

## 📊 Resultados

- Acurácia inicial: **~75%**
- Após aumento de dados: **~85%**
- Modelo mostrou bom desempenho principalmente em imagens com maior clareza visual da ação.

## 🧪 Como Testar

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
