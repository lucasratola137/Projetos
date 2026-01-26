from platform import python_version

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#define uma constante com o caminho do arquivo de dados
file_path = 'dados_clientes.csv'
#carrega o dataset
data_set = pd.read_csv(file_path)

#resumo estatístico dos dados (das 3 colunas que serão usadas na clusterização)
data_set[['idade','renda_anual','pontuacao_gastos']].describe()

#padronização dos dados - escala z-score dos dados
padronizar = StandardScaler()
dados_padronizados = padronizar.fit_transform(data_set[['idade','renda_anual','pontuacao_gastos']])

#demonstração a via de comparação dos dados originais e padronizados - sem o índice e apenas as 10 primeiras linhas
print(data_set[['idade','renda_anual','pontuacao_gastos']].head(10).to_string(index=False))
print(dados_padronizados[:10])

#define o número de clusters / segmentos
k = 3
#cria o modelo de KMeans
kmeans = KMeans(n_clusters = k)
#treina o modelo
kmeans.fit(dados_padronizados)
data_set['cluster'] = kmeans.labels_
#mostra as 10 primeiras linhas do dataset com os clusters atribuídos
print(data_set.head(10).to_string(index=False))
#salva o novo arquivo com os segmentos
data_set.to_csv('dados_clientes_segmentados.csv', index=False)
