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
#from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
#from tensorflow.keras.applications.inception_v3 import preprocess_input
#from tensorflow.keras.applications.densenet import preprocess_input


# Aqui estou carregando o modelo de rede neural desejada.
# Utilize apenas a rede desejada o restante deixe como comentário
#model = ResNet152V2(weights='imagenet', include_top=False)
model = VGG16(weights='imagenet', include_top=False)
#model = InceptionV3(weights='imagenet', include_top=False)
#model = DenseNet201(weights='imagenet', include_top=False)

model.summary()
list_features_extracted = []

file_list = glob.glob("images/*.jpg")
file_list = sorted(file_list)

for num, file in enumerate(file_list):

    if num % 5 == 0:
        print('Class: ')

    print(' - ', file)

    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = numpy.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = numpy.array(model.predict(x), dtype='float16')
    list_features_extracted.append(features.flatten())

y_pred = KMeans(n_clusters=5, random_state=200).fit_predict(list_features_extracted)

print(y_pred)
