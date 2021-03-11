
# Biblioteca para leitura de CSV
from pandas import read_csv

# Carregamento dos dados da tabela
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Carregamento dos dados da tabela
dados_tabela = read_csv('tabela/dataset_iris.csv', usecols=[1, 2, 3, 4], engine='python')
rotulos = read_csv('tabela/dataset_iris.csv', usecols=[0], engine='python')
conjunto_de_dados = dados_tabela.values
rotulos = rotulos.values

# Conversão para float32
conjunto_de_dados = conjunto_de_dados.astype('float32')
rotulos = rotulos.astype('int32')

# recorte dos dados
dados_para_predicao = conjunto_de_dados[0:len(conjunto_de_dados), :]
rotulos = rotulos[0:len(rotulos), :]


# Aqui estão alguns algoritmos de clustering mais conhecidos
# Para utilizar um deles basta remover o comentário do algoritmo desejado e comentar os demais
resultados = KMeans(n_clusters=3, random_state=10).fit_predict(dados_para_predicao)
#resultados = MeanShift(0.8).fit_predict(dados_para_predicao)
#resultados = AgglomerativeClustering(n_clusters=3, affinity='euclidean').fit_predict(dados_para_predicao)
#resultados = DBSCAN(0.7).fit_predict(dados_para_predicao)
#resultados = OPTICS(min_samples=20, xi=.05, min_cluster_size=.05).fit_predict(dados_para_predicao)
#resultados = GaussianMixture(n_components=3, covariance_type='full').fit_predict(dados_para_predicao)

for i in range(len(resultados)):

    print('Flor id: ', rotulos[i][0], " predicao: ", resultados[i])