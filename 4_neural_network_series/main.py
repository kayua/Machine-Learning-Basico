# Importando bibliotecas necessárias

# Biblioteca para listagem de arquivos
import glob

# Biblioteca numérica
import numpy

# Biblioteca de aprendizado de máquinas sklearn import do KMeans
from sklearn.cluster import KMeans

# Importando Application ResNet152V2, VGG16, InceptionV3, DenseNet201
# Elas serão baixadas automáticament antes da execução(demora um pouco)
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201


# Importando preprocessing para carregar entradas
from tensorflow.keras.preprocessing import image
# Importando preprocessador para adptar imagens a entrada da rede
# Para cada uma das redes, use apenas o preprocess_input específico o restante deixe como comentário
from tensorflow.keras.applications.resnet_v2 import preprocess_input
#from tensorflow.keras.applications.vgg16 import preprocess_input
#from tensorflow.keras.applications.inception_v3 import preprocess_input
#from tensorflow.keras.applications.densenet import preprocess_input


# Aqui estou carregando o modelo de rede neural desejada.
# Utilize apenas a rede desejada o restante deixe como comentário
# Imagenet é uma base de dados com muitas imagens, algumas das quais foram usadas para pre-treinar esse modelo
# O include_top é uma espécie de camada extra no topo da rede. Não recomendado. Evite problemas com os modelos
model = ResNet152V2(weights='imagenet', include_top=False)
#model = VGG16(weights='imagenet', include_top=False)
#model = InceptionV3(weights='imagenet', include_top=False)
#model = DenseNet201(weights='imagenet', include_top=False)

# Summary serve para mostras a rede neural detalhadamente. Opcional
model.summary()
# Lista que armazenará as features extraidas
lista_de_features_extraidas = []

# Aqui estou listando e ordenando as imagens do diretório images/
lista_nome_arquivos = glob.glob("images/*.jpg")
lista_nome_arquivos = sorted(lista_nome_arquivos)

# Esse laço serve para extrair features de todas imagens
for indice, diretorio_image in enumerate(lista_nome_arquivos):

    if indice % 5 == 0:
        print('Class: ')

    print(' - ', diretorio_image)

    # Abrindo a imagem usando uma ferramenta do tensorflow. Dá para usar outras ferramentas sem problema
    imagem_aberta = image.load_img(diretorio_image, target_size=(224, 224))

    # Transformando a imagem em um array
    imagem_array = image.img_to_array(imagem_aberta)

    # Tranformando a imagem em um array numpy "expandido"(Útil quando houver lote de imagens, necessário para o pré...)
    imagem_array = numpy.expand_dims(imagem_array, axis=0)

    # Adaptação da imagem para entrada da rede neural
    imagem_array = preprocess_input(imagem_array)

    # Extração da feature e conversão para numpy array float16
    features = numpy.array(model.predict(imagem_array), dtype='float16')

    # Tranformando saída para um vetor. A predição normamente gera uma "matriz". Para o Kmeans é necessário um "vetor"
    # Guardando features extraídas
    lista_de_features_extraidas.append(features.flatten())

# Instânciando o Kmeans e fazendo o aprendizado não supervisionado, ele já faz a predição dos resultados
# Kmeans(<número de classes>, <número de repetições>)
rotulos_preditos = KMeans(n_clusters=5, random_state=100).fit_predict(lista_de_features_extraidas)

print(rotulos_preditos)
