# Importando bibliotecas necessárias


# Biblioteca numérica
import numpy

# Visualisar imagens
from matplotlib import pyplot

# Biblioteca para leitura de CSV
from pandas import read_csv

# Uma das inúmeras funções do SKlear úteis para limpar o dataset
from sklearn.preprocessing import MinMaxScaler
# Importação de camadas do Keras

from keras.models import Sequential

# Dense é a rede "tradicional", neurônios simples
from keras.layers import Dense

# Recurrent é a classe mais simples de redes neurais recorrentes.
from keras.layers import recurrent

# LSTM é uma classe de redes neurais recorrentes mais avençadas que as RNN comuns
from keras.layers import LSTM


# Função usada para gerar o conjunto de dados da serie
from tensorflow import initializers


def create_dataset(dados_da_serie):
    # Lista de amostras
    entrada_rede_neural_X, saida_rede_neural_Y = [], []

    # Para cada ponto dos dados no tempo, adiciono a lista o valor no ponto (Tanto para X quanto para Y)
    # Isso ocorre porque a partir do ponto anterior eu devo predizer exatamente o valor posterior.
    for i in range(len(dados_da_serie) - 2):

        dado_em_um_determinado_ponto = dados_da_serie[(i + 1), 0]
        entrada_rede_neural_X.append([dado_em_um_determinado_ponto])
        saida_rede_neural_Y.append(dado_em_um_determinado_ponto)

    return entrada_rede_neural_X, saida_rede_neural_Y

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

# Carregamento dos dados da tabela
dados_tabela = read_csv('tabela/tabela.csv', usecols=[1], engine='python')
conjunto_de_dados = dados_tabela.values

# Conversão para float32
conjunto_de_dados = conjunto_de_dados.astype('float32')


# Separação os dados em dois conjuntos, teste(40%) e trainamento(60%)
tamanho_conjunto_treinamento = int(len(conjunto_de_dados) * 0.60)
tamanho_conjunto_testes = len(conjunto_de_dados) - tamanho_conjunto_treinamento
dados_para_treinamento = conjunto_de_dados[0:tamanho_conjunto_treinamento, :]
dados_para_testes = conjunto_de_dados[tamanho_conjunto_treinamento:len(conjunto_de_dados), :]

# Criando o dataset trainamento e teste
conjunto_de_trainamento_entrada_rede, conjunto_treinamento_saida_rede = create_dataset(dados_para_treinamento)
testX, testY = create_dataset(dados_para_testes)

# Convertendo lista de amostras para numpy array
conjunto_de_trainamento_entrada_rede = numpy.array(conjunto_de_trainamento_entrada_rede)
conjunto_treinamento_saida_rede = numpy.array(conjunto_treinamento_saida_rede)
testX = numpy.array(testX)
testY = numpy.array(testY)


conjunto_de_trainamento_entrada_rede = numpy.reshape(conjunto_de_trainamento_entrada_rede, (conjunto_de_trainamento_entrada_rede.shape[0], 1, conjunto_de_trainamento_entrada_rede.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
inicializador = initializers.RandomNormal(stddev=0.01)
model.add(LSTM(18, input_shape=(1, 1), kernel_initializer=inicializador))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(conjunto_de_trainamento_entrada_rede, conjunto_treinamento_saida_rede, epochs=2600, batch_size=1)
# make predictions
trainPredict = model.predict(conjunto_de_trainamento_entrada_rede)
testPredict = model.predict(testX)
# invert predictions
# calculate root mean squared error
trainPredictPlot = numpy.empty_like(conjunto_de_dados)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[1:len(trainPredict) + 1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(conjunto_de_dados)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + 2 + 1:len(conjunto_de_dados) - 1, :] = testPredict
# plot baseline and predictions
pyplot.plot(conjunto_de_dados)
pyplot.plot(testPredictPlot)
pyplot.show()
