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
from keras.layers import Dense, Input

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
    pyplot.title('Erro médio absoluto')
    # Plota os valores de acurácia da rede em cada época.
    pyplot.plot(modelo.history['mean_absolute_error'], color='blue')

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
conjunto_de_testes_entrada_rede, conjunto_de_testes_saida_rede = create_dataset(dados_para_testes)

# Convertendo lista de amostras para numpy array
conjunto_de_trainamento_entrada_rede = numpy.array(conjunto_de_trainamento_entrada_rede)
conjunto_treinamento_saida_rede = numpy.array(conjunto_treinamento_saida_rede)
conjunto_de_testes_entrada_rede = numpy.array(conjunto_de_testes_entrada_rede)
conjunto_de_testes_saida_rede = numpy.array(conjunto_de_testes_saida_rede)

# Ajusta o formato a entrada da rede
conjunto_de_trainamento_entrada_rede = numpy.reshape(conjunto_de_trainamento_entrada_rede, (conjunto_de_trainamento_entrada_rede.shape[0], 1, conjunto_de_trainamento_entrada_rede.shape[1]))
conjunto_de_testes_entrada_rede = numpy.reshape(conjunto_de_testes_entrada_rede, (conjunto_de_testes_entrada_rede.shape[0], 1, conjunto_de_testes_entrada_rede.shape[1]))

# Instância a rede neural
model = Sequential()

# Inicializa os pesos usando distribuição normal com desvio stddev de 0.01
# Fiz isso, pois a rede estava demorando muito para treinar. LSTM normalmente demora
inicializador = initializers.RandomNormal(stddev=0.01)

# Definindo formato de entrada da rede
model.add(Input(shape=(1, 1)))

# LSTM(<Número de células>, <Kernel de inicialização>(Opcional))
model.add(LSTM(18, kernel_initializer=inicializador))

# Dense é a rede "tradicional", neurônios simples.
model.add(Dense(16))
model.add(Dense(1))

# Compilando a rede neural, aqui utilizei Erro médio quadrático pois meu objetivo é minimizar a diferença entre
# os resultado preditos e o resultado original
# O otimizador, utilizei o gradiente estocástico. Entre os otimizadores é o que acho mais tranquilo de usar.
# Não lembro de ter tido problemas com ótimos locais usando esse método de aproximação
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# metrica utilizei o erro absoluto médio
model.compile(loss='mean_squared_error', optimizer='adam', metrics="mean_absolute_error")

resultados = model.fit(conjunto_de_trainamento_entrada_rede, conjunto_treinamento_saida_rede, epochs=2800, batch_size=1)

# Predição da rede neural
resultados_preditos_treinamento = model.predict(conjunto_de_trainamento_entrada_rede)
resultados_predictos_testes = model.predict(conjunto_de_testes_entrada_rede)

# Usado para formatar a saída para o plot
resultados_formatados_plot_treinamento = numpy.empty_like(conjunto_de_dados)
resultados_formatados_plot_treinamento[:, :] = numpy.nan
resultados_formatados_plot_treinamento[1:len(resultados_preditos_treinamento) + 1, :] = resultados_preditos_treinamento

resultados_formatados_plot_predicoes = numpy.empty_like(conjunto_de_dados)
resultados_formatados_plot_predicoes[:, :] = numpy.nan
resultados_formatados_plot_predicoes[len(resultados_preditos_treinamento) + 2 + 1:len(conjunto_de_dados) - 1, :] = resultados_predictos_testes

pyplot.title('(Dataset Original- Azul)/(Predição da rede neural- Amarelo) ')
pyplot.plot(conjunto_de_dados)
pyplot.plot(resultados_formatados_plot_predicoes)
pyplot.show()

visualizar_resultados(resultados)
