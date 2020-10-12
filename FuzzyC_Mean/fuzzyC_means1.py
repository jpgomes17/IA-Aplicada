
import skfuzzy as fuzz
import numpy as np
from sklearn.datasets import load_wine
from sklearn import metrics
from scipy.spatial import distance



#Carrega o wine dataset em wine
wine = load_wine()
#Separando dados para treinamento
dados1 = wine.data[0:30]
dados2 = wine.data[59:94]
dados3 = wine.data[130:154]
dados_treinamento = np.vstack([dados1, dados2, dados3])
alldata = dados_treinamento.transpose()
label = wine.target
ncenters = 3

#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, pit, fpc = fuzz.cluster.cmeans(
    alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)


#Separando dados para teste
dados4 = wine.data[30:59]
dados5 = wine.data[94:130]
dados6 = wine.data[154:178]
dados_teste = np.vstack([dados4, dados5, dados6])
newdata = dados_teste

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2,  error=0.005, maxiter=1000)
cluster_membership_predict = np.argmax(u, axis=0)
c = cluster_membership_predict
print(c)

#Analise de desempenho
cont = 0
for i in range(29):
    if c[i] == 0:
        cont = cont + 1

for i in range(29,65):
    if c[i] == 2:
        cont = cont + 1

for i in range(65,89):
    if c[i] == 1:
        cont = cont + 1

'índice de desempenho'
indice = (cont/c.size)*100

print(f"N iteracoes: {pit}")
#print(f"Sum das distancias: {kmeans.inertia_}")
print(f"% de acertos: {indice}")

#Score de similaridade
dados7 = wine.target[30:59]
dados8 = wine.target[94:130]
dados9 = wine.target[154:178]
labels_true = np.hstack([dados7, dados8, dados9])
print(labels_true)
print(c)
a = metrics.adjusted_rand_score(labels_true, c)
print(f"score de similaridade: {a}")


#
# def caldist(cntr, X, Y):
#     sumDist = 0
#     center= 0
#     for i in X:
#         if Y[i] == 0:
#             center = cntr[0]
#         elif Y[i] == 1:
#             center = cntr[1]
#         if Y[i] == 2:
#             center = cntr[2]
#         sumDist += distance(i, center)
#
#     return sumDist
