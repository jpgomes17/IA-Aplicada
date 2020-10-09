# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:17:51 2020

@author: jpgom
"""

print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import load_iris

'''
#Carrega o iris dataset em iris 
iris = load_iris() 
alldate = iris.data
'Separando dados para treinamento'
dados1 = iris.data[0:25]
dados2 = iris.data[50:75]
dados3 = iris.data[100:125]
dados_treinamento = np.vstack([dados1, dados2, dados3])

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(dados_treinamento, quantile=0.2, n_samples=75)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(dados_treinamento)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

#Prevendo novos valores
'Separando dados para teste'
dados4 = iris.data[25:50]
dados5 = iris.data[75:100]
dados6 = iris.data[125:150]
dados_teste = np.vstack([dados4, dados5, dados6])
c = ms.predict(dados_teste)
print(c)

cont = 0
for i in range(25):
    if c[i] == 1:
        cont = cont + 1
        
for i in range(25,50):
    if c[i] == 0:
        cont = cont + 1
        
for i in range(50,75):
    if c[i] == 2:
        cont = cont + 1

'índice de desempenho'
indice = (cont/c.size)*100
print(indice)
'''

from sklearn.datasets import load_wine

#Carrega o wine dataset em wine 
wine = load_wine() 
alldate = wine.data



'Separando dados para treinamento'
dados1 = wine.data[0:30]
dados2 = wine.data[59:94]
dados3 = wine.data[130:154]
dados_treinamento = np.vstack([dados1, dados2, dados3])

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(wine.data, quantile=0.2, n_samples=178)

ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
ms.fit(dados_treinamento)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

#Prevendo novos valores
'Separando dados para teste'
dados4 = wine.data[30:59]
dados5 = wine.data[94:130]
dados6 = wine.data[154:178]
dados_teste = np.vstack([dados4, dados5, dados6])
c = ms.predict(dados_teste)
print(c)

cont = 0
for i in range(29):
    if c[i] == 2:
        cont = cont + 1
        
for i in range(29,65):
    if c[i] == 0:
        cont = cont + 1
        
for i in range(65,89):
    if c[i] == 1:
        cont = cont + 1

'índice de desempenho'
indice = (cont/c.size)*100
print(indice)
