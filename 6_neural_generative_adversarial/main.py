# Importando bibliotecas necessárias

# Biblioteca para listagem de arquivos

# Atenção, esse algoritmo demora algumas horas para treinar

from glob import glob

# Biblioteca para abrir imagens
import cv2.cv2

# Bibliote numérica
import numpy

# Convolução bidimensional
from keras.layers import Conv2D

# Activations do keras: [relu - softmax - LeakyReLU - prelu - elu - thresholdedrelu]
from keras.layers import Activation, MaxPooling2D

# Dense é a rede "tradicional", neurônios simples
from keras.layers import Dense

# Usado para achatar saídas ou entradas em rede neurais 2D->1D [[1.4, 1.1], [1.8, 3.1]] -> [1.4, 1.1, 1.8, 3.1]
from keras.layers import Flatten

# Layers para definir o formato da rede neural
from keras.layers import Input

# Camada capaz de redimensionar o formato sa saída de uma layer
from keras.layers import Reshape

# Camada de pooling médio reduz a dimensão
from keras.layers import AveragePooling2D

# Camada que amplia um imagem
from keras.layers import UpSampling2D

# Camada de preenchimento com zeros
from keras.layers import ZeroPadding2D

# Classe modelo para as rede neurais(Pode ser feito na mão caso achem melhor)
from keras.models import Sequential
from keras.models import Model

# Função de "atualização" dos pesos na rede neural("Perda"), é necessária para o treinamento
from keras.optimizers import Adam

# Criar barrinha de carregamento de imagens
from tqdm import tqdm

# Importando tensorflow
import tensorflow as tf


# Função responsável por criar o gerador imagens,
# que a partir do ruído aleatório(pontos de latência) irá criar a imagem fake
def funcao_criar_gerador_image():
    # Instância do model
    model = Sequential()

    # Camada de densa responsável por receber os ruídos
    # Cada camada é ativada com uma relu
    model.add(Dense(6 * 9 * 256, input_dim=200))

    # Camade de transformação, recebe 13824 saídas da camada densa
    # e devolve 256 imagens (6 x 9)
    model.add(Reshape((6, 9, 256)))

    # Camadas convolutivas, a partir daqui o padrão será:

    # Conv2D(Filtros, tamanho_filtros, Normalização)
    # UpSampling2D() <- Aumenta o tamanho da imagem
    # ZeronPadding2D <- Preenchimento
    # A ideia principal é aumentar de forma progressiva a saida até o tamanho máximo da imagem

    model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
    model.add(UpSampling2D())
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))

    model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
    model.add(UpSampling2D())
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))

    model.add(Conv2D(64, kernel_size=(3, 3), padding="same"))
    model.add(UpSampling2D())
    model.add(ZeroPadding2D(padding=(0, (1, 0))))

    model.add(Conv2D(32, kernel_size=(3, 3), padding="same"))
    model.add(UpSampling2D())

    model.add(Conv2D(32, kernel_size=(3, 3), padding="same"))
    model.add(UpSampling2D())

    model.add(Conv2D(16, kernel_size=(3, 3), padding="same"))

    model.add(Conv2D(3, kernel_size=(1, 1), padding="same"))
    model.add(Activation("tanh"))

    # Aqui eu ligo a entrada do modelo, ao modelo em si
    entrada_ruidos = Input(shape=(200,))
    saida_modelo = model(entrada_ruidos)

    return Model(entrada_ruidos, saida_modelo)


def funcao_criar_discriminador_imagem():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(200, 300, 3), padding="same"))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    entrada_image = Input(shape=(200, 300, 3))
    validadecao_model = model(entrada_image)

    return Model(entrada_image, validadecao_model)

# Essa função serve para converter a imagem para um formato compatível com TensorFlow.
# Na prática ela abre a imagem, decodifica, converte para float e redimensiona a imagem.


def converte_image_formato_tensorflow(nome_arquivo):

    image = tf.io.read_file(nome_arquivo)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [200, 300]) / 255.0
    return image


optimizer = Adam(0.001, 0.5)

discriminador = funcao_criar_discriminador_imagem()
gerador = funcao_criar_gerador_image()
discriminador.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminador.trainable = False
z = Input(shape=(200,))
img = gerador(z)
valid = discriminador(img)

rede_gan = Model(z, valid)
rede_gan.compile(loss='binary_crossentropy', optimizer=optimizer)

dos = glob("drive/MyDrive/dataset/*")

image_list = []

for i in tqdm(dos):
    image_list.append(converte_image_formato_tensorflow(i))

X_train = numpy.asarray(image_list, dtype="int32")

ones = numpy.ones((30, 1))
zeros = numpy.zeros((30, 1))

epoch = 0
noise = numpy.random.normal(0, 1, (30, 200))
idx = numpy.random.randint(0, X_train.shape[0], 30)

while (1):

    epoch += 1

    imgs = X_train[idx]

    gen_imgs = gerador.predict(noise)
    d_loss_r = discriminador.train_on_batch(imgs, ones)
    d_loss_f = discriminador.train_on_batch(gen_imgs, zeros)
    d_loss = numpy.add(d_loss_r, d_loss_f) * 0.5

    g_loss = rede_gan.train_on_batch(noise, ones)

    if epoch % 100 == 0:
        rand_noise = numpy.random.normal(0, 1, (1, 200))
        pred = gerador.predict(rand_noise)
        confidence = discriminador.predict(pred)
        gen_img = (0.5 * pred[0] + 0.5) * 255

        print("%d D loss: %f, acc.: %.2f%% G loss: %f" % (epoch, d_loss[0], 100 * d_loss[1], g_loss / 10))
        cv2.imwrite('drive/MyDrive/imagens/image_saida' + str(epoch) + '.png', gen_img)

    if epoch == 1000000:
        break
