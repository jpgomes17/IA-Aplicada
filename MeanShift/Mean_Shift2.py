import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import load_wine
from sklearn import metrics
import skfuzzy as fuzz
#Carrega o wine dataset em wine
wine = load_wine()

#Separando dados para treinamento
dados1 = wine.data[0:50]
dados2 = wine.data[59:115]
dados3 = wine.data[130:165]
dados_treinamento = np.vstack([dados1, dados2, dados3])

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(dados_treinamento, quantile=0.2, n_samples=89)

ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
ms.fit(dados_treinamento)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

#Prevendo novos valores
'Separando dados para teste'
dados4 = wine.data[50:59]
dados5 = wine.data[115:130]
dados6 = wine.data[165:178]
dados_teste = np.vstack([dados4, dados5, dados6])
c = ms.predict(dados_teste)
print(c)

#analise de desempenho
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


'índice de desempenho'
indice = (cont/c.size)*100


print(f"N iteracoes: {ms.n_iter_}")

print(f"% de acertos: {indice}")

#Score de similaridade
dados7 = wine.target[50:59]
dados8 = wine.target[115:130]
dados9 = wine.target[165:178]
labels_true = np.hstack([dados9, dados8, dados7])
print(labels_true)
print(c)
a = metrics.adjusted_rand_score(labels_true, c)
print(f"score de similaridade: {a}")
