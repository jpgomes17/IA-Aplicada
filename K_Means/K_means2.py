from sklearn.cluster import KMeans
import numpy as np

from collections import Counter
from sklearn.datasets import load_wine
from sklearn import metrics

#carrega dataset
wine = load_wine()
#y = wine.target[:]
#print(y)
print(Counter(y))

#Separando dados para treinamento
dados1 = wine.data[0:50]
dados2 = wine.data[59:115]
dados3 = wine.data[130:165]
dados_treinamento = np.vstack([dados1, dados2, dados3])

#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=1).fit(dados_treinamento)
kmeans.labels_
kmeans.cluster_centers_

#Prevendo novos valores
dados4 = wine.data[50:59]
dados5 = wine.data[115:130]
dados6 = wine.data[165:178]
dados_teste = np.vstack([dados4, dados5, dados6])
c = kmeans.predict(dados_teste)
#Valores previstos para os intervalos dados4, dados5, dados6
print(c)
#print(len(c))
#print(Counter(c))

#Analise de desempenho
cont = 0
for i in range(9):
    if c[i] == 2:
        cont = cont + 1
for i in range(9,24):
    if c[i] == 0:
        cont = cont + 1

for i in range(24,37):
    if c[i] == 1:
        cont = cont + 1


#índice de desempenho
indice = (cont/c.size)*100



print(f"N iteracoes: {kmeans.n_iter_}")

print(f"Sum das distancias: {kmeans.inertia_}")
print(f"% de acertos: {indice}")

#score de similaridade
dados7 = wine.target[50:59]
dados8 = wine.target[115:130]
dados9 = wine.target[165:178]
labels_true = np.hstack([dados9, dados8, dados7])
print(labels_true)
print(c)
a = metrics.adjusted_rand_score(labels_true, c)
print(f"score de similaridade: {a}")
