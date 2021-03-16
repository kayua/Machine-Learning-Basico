# Importando bibliotecas necessárias

# Visualisar imagens
from matplotlib import pyplot

# Dataset das figuras.
from keras.datasets import cifar10

# Ferramenta para converter valores em códigos [1,2 3] -> [[1,0,0], [0,1,0], [0, 0, 1]] Ex. [1] -> [1, 0, 0]
from keras.utils import to_categorical

# Classe modelo para as rede neurais(Pode ser feito na mão caso achem melhor)
from keras.models import Sequential

# Camadas para rede neural

# Convolução bidimensional
from keras.layers import Conv2D

# Pooling máximo reduz o tamanho de sua entrada(Bidimensional) [[3,4],[5,1]] (MaxPooling2D(2,2)) -> [5]
from keras.layers import MaxPooling2D

# Dense é a rede "tradicional", neurônios simples
from keras.layers import Dense

# Usado para achatar saídas ou entradas em rede neurais 2D->1D [[1.4, 1.1], [1.8, 3.1]] -> [1.4, 1.1, 1.8, 3.1]
from keras.layers import Flatten

# Função de "atualização" dos pesos na rede neural("Perda"), é necessária para o treinamento
from keras.optimizers import SGD


# Obtendo dataset para treinamento - Aqui estou usando um dataset presete na biblioteca
# Para uma rede neural é necessário ter dois conjuntos de dados.
# Uma para efetuar o treinamento e outro para testes.
# Nesse caso, eu estou recebendo duas tuplas(X,Y) para cada um dos conjuntos(samples_training e samples test)

# Imagens presentes no dataset

#   0: airplane
#   1: automobile
#   2: bird
#   3: cat
#   4: deer
#   5: dog
#   6: frog
#   7: horse
#   8: ship
#   9: truck

# Função simples para pegar os dados do dataset
from tensorflow.python.keras import Input


def carregar_dataset():

    # Separar o dataset de trainamento do dataset de test
    (entrada_treinamento_X, saida_treinamento_Y), (entrada_testes_X, saida_testes_Y) = cifar10.load_data()

    # Converte o código decimal da classe em um vetor Ex. [1] -> [1, 0, 0, 0, 0, 0, 0, 0]
    saida_treinamento_Y = to_categorical(saida_treinamento_Y)

    # Converte o código decimal da classe em um vetor Ex. [1] -> [1, 0, 0, 0, 0, 0, 0, 0]
    saida_testes_Y = to_categorical(saida_testes_Y)

    return entrada_treinamento_X, saida_treinamento_Y, entrada_testes_X, saida_testes_Y


# Essa função converte os valores do dataset para float32(Redes neurais trabalham com pontos flutuantes do numpy)

def converter_para_float32(amostra_trainamento, amostra_testes):
    # Conversão a float32
    imagens_normalizadas_treino = amostra_trainamento.astype('float32')
    imagens_normalizadas_tests = amostra_testes.astype('float32')

    # Normalização entre 0-255 (Imagem com 256 valores possíveis por pixel)
    imagens_normalizadas_treino = imagens_normalizadas_treino / 255.0
    imagens_normalizadas_tests = imagens_normalizadas_tests / 255.0

    return imagens_normalizadas_treino, imagens_normalizadas_tests


def visualizar_resultados(modelo):

    pyplot.subplot(211)
    pyplot.title('Perda da rede(treinamento)')

    # Plota os valores de perda da rede em cada época.
    pyplot.plot(modelo.history['loss'], color='blue')

    pyplot.subplot(212)
    pyplot.title('Acurácia da classificação')
    # Plota os valores de acurácia da rede em cada época.
    pyplot.plot(modelo.history['accuracy'], color='blue')

    pyplot.show('figura')
    pyplot.close()

# Obtem as amostras do dataset
trainX, trainY, testX, testY = carregar_dataset()

# Converte os valores interios para float32
trainX, testX = converter_para_float32(trainX, testX)

# Instância uma rede neural
model = Sequential()

# Define o tamanho dos dados de entrada. Ex (Imagens 32x32 com 3 cores)
model.add(Input(shape=(32, 32, 3)))

# Típica rede neural convolutiva

# Camada convolutiva com 32 filtros com tamanho (3x3)
model.add(Conv2D(32, (3, 3)))

# Camada de pooling que reduz o tamanho da imagem de Ex. 32x32 para 16x16. Tamanho do pooling divide a imagem 32/2 = 16
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3)))

# Transforma o formato de entrada(32, 32, 3) em um vetor continuo(3072).
# Essa camada é necessária sempre que houver uma entrada com mais de uma dimensão entrando em uma camada densa
model.add(Flatten())

# Camada densa, layer mais simples. É um uma layer composta por neurônios
# Dense(<NUMERO DE NEURÔNIO>, activation=<NOME DA FUNÇÂO DE ATIVAÇÃO>, +Argumentos) # O argumento activation é opcional

# Activations do keras: [relu - softmax - LeakyReLU - prelu - elu - thresholdedrelu]
# Activations: https://keras.io/api/layers/activation_layers/

# Outras camadas pode ser intercaladas com essas, porém é necessário verificar se o tamanho das saídas são compatíveis
# Importante: A última camada de definirá o tamanho de saída. Nesse caso, o tamanho será 10.
# Dica: Não use muitas camadas, isso pode acarretar em demora no treinamento. O gradinte é calculado pela composta
# sucessiva, desde a camada de saída até a camada de entrada(Propagação para trás). Assim, o valor da deriva tende
# progressivamente a zero, log

model.add(Dense(10, activation='softmax'))

# Summary serve para mostras a rede neural detalhadamente. Opcional
model.summary()

# Definição do otimizador. Nesse caso, foi escolhido o SGD (Gradiente descendente) usado pelo Multilayer Perceptron

# Optimizer do keras:  [SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
# lr = Taxa de aprendizado. É um valor que multiplica o valor de ajuste do peso.
# Valores altos de lr podem dificultar o treinamento, valores muito baixos tornam a aprendizagem demorada.
# Momentum é uma forma de acelerar o treinamento.
# Momentum https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/

opt = SGD(lr=0.01, momentum=0.9)

# O modelo precisa ser compilado, para isso chamamos compile()


# Metrics é utilizado para avaliar seu modelo.

#Accuracy metrics: accuracy, binary_accuracy, categorical_accuracy, top_K_categorical_accuracy, etc..

#Probabilistic metrics: binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, poisson

#Regression metrics: mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# compile(<optimizer>, <loss>, <metrics>)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# treinar o modelo: Para treinar
# fit(<Entradas X>,<Saida Y>,<Numero Épocas>, <Outros argumentos. Ex. Verbose, batch_size, etc...> )
# Épocas = Número de ciclos de treinamento
# x= Entradas dos dados: (Imagem Cat (32x32x3))
# y= Saída desejada: ([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) <- Código equivalente ao gato.
resultados_treinamento = model.fit(x=trainX, y=trainY, epochs=100)

# Predição dos resultados: Adicionei uma lista de entradas para teste que havia reservado
resultado_predito = model.predict(testX)

print("Imagem rótulo original: ", testY[10], "\nRótulo predito pela rede: ", resultado_predito[10])

visualizar_resultados(resultados_treinamento)# Função para visualizar o erro diminuindo da rede

# Caso queiram salvar o seus modelos de rede neural:
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/