from glob import glob
from random import random, randrange, randint

import cv2.cv2
import numpy

import numpy as np

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import ZeroPadding2D

from keras.models import Model, Sequential
from keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm


def create_generator():
    model = Sequential()
    model.add(Dense(6 * 9 * 256, activation="relu", use_bias=False, input_dim=100))
    model.add(Reshape((6, 9, 256)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(ZeroPadding2D(padding=(0, (1, 0))))
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(16, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=(1, 1), padding="same"))
    model.add(Activation("tanh"))
    noise = Input(shape=(100,))
    img = model(noise)
    model.summary()
    return Model(noise, img)


def create_discriminator():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(100, 150, 3), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(AveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(AveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    img = Input(shape=(100, 150, 3))
    validity = model(img)
    model.summary()

    return Model(img, validity)


import tensorflow as tf

IMAGEWIDTH = 100
IMAGEHEIGHT = 100
CHANNEL = 3
EPOCHS = 10


def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [100, 150]) / 255.0
    return image


optimizer = Adam(0.0002, 0.5)

discriminador = create_discriminator()
gerador = create_generator()
discriminador.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminador.trainable = False
z = Input(shape=(100,))
img = gerador(z)
valid = discriminador(img)

rede_gan = Model(z, valid)
rede_gan.compile(loss='binary_crossentropy', optimizer=optimizer)

dirs = glob("dataset_gan/*")
dos = []
list_latences = []

for num, i in enumerate(dirs):

    dos.extend(glob(i + '/*'))

    for j in range(len(glob(i + '/*'))):

        random_list = []

        for k in range(90):

            random_list.append(randrange(0, 1))

        list_latences.append(numpy.array(to_categorical(num, 10).tolist()+random_list))


image_list = []

for i in tqdm(dos):
    image_list.append(parse_image(i))

X_train = numpy.asarray(image_list, dtype="int32")
print('Treinando')
ones = np.ones((128, 1))
zeros = np.zeros((128, 1))

epoch = 0

while (1):

    epoch += 1
    idx = randint(1, X_train.shape[0])
    imgs = X_train[idx]
    noise = list_latences[idx]

    gen_imgs = gerador.predict(noise)
    d_loss_r = discriminador.train_on_batch(imgs, ones)
    d_loss_f = discriminador.train_on_batch(gen_imgs, zeros)
    d_loss = np.add(d_loss_r, d_loss_f) * 0.5

    g_loss = rede_gan.train_on_batch(noise, ones)

    print("%d D loss: %f, acc.: %.2f%% G loss: %f" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
    rand_noise = np.random.normal(0, 1, (1, 100))
    pred = gerador.predict(rand_noise)
    confidence = discriminador.predict(pred)

    gen_img = (0.5 * pred[0] + 0.5) * 255

    if epoch % 10 == 0:
        cv2.imwrite('imagens/image_saida' + str(epoch) + '.png', gen_img)

    if epoch == 100:
        break
