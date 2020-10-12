
import skfuzzy as fuzz
import numpy as np
from sklearn.datasets import load_wine
from sklearn import metrics
from scipy.spatial import distance



#Carrega o wine dataset em wine
wine = load_wine()
'Separando dados para treinamento'
dados1 = wine.data[0:50]
dados2 = wine.data[59:115]
dados3 = wine.data[130:165]
dados_treinamento = np.vstack([dados1, dados2, dados3])
alldata = dados_treinamento.transpose()
label = wine.target
ncenters = 3
#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, pit, fpc = fuzz.cluster.cmeans(
    alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)


# Generate uniformly sampled data spread across the range [0, 10] in x and y
'Separando dados para teste'
dados4 = wine.data[50:59]
dados5 = wine.data[115:130]
dados6 = wine.data[165:178]
dados_teste = np.vstack([dados4, dados5, dados6])
newdata = dados_teste

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.0000000001, maxiter=1000)
cluster_membership_predict = np.argmax(u, axis=0)
c = cluster_membership_predict
print(c)

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
print(indice)

print(f"N iteracoes: {pit}")
#print(f"Sum das distancias: {kmeans.inertia_}")
print(f"% de acertos: {indice}")

#Separando dados para cálculo de desempenho
dados7 = wine.target[50:59]
dados8 = wine.target[115:130]
dados9 = wine.target[165:178]
labels_true = np.hstack([dados7, dados8, dados9])
print(labels_true)
print(c)
a = metrics.adjusted_rand_score(labels_true, c)
print(f"score de similaridade: {a}")
print(cntr)

# from sklearn import metrics
# list1 = "0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2"
# list1 = list(list1.replace(" ",""))
# list3=[]
# for item in list1:
#     if item == '0':
#         list3.append(2)
#     elif item == '1':
#         list3.append(0)
#     elif item == '2':
#         list3.append(1)
# print(list3)
#


# list2 = "2 2 2 2 2 2 1 2 2 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 0"
# list2 = list(list2.replace(" ",""))
# list4=[]
# for item in list2:
#     if item == '0':
#         list4.append(0)
#     elif item == '1':
#         list4.append(1)
#     elif item == '2':
#         list4.append(2)
# print(list4)
# a = metrics.adjusted_rand_score(list4, list3)
# print(f"score de similaridade: {a}")
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
