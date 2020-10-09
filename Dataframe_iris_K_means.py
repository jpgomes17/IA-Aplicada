# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from sklearn.cluster import KMeans
#from sklearn.datasets import load_iris
import numpy as np

''' 
#Carrega o iris dataset em iris 
iris = load_iris() 
alldate = iris.data



'Separando dados para treinamento'
dados1 = iris.data[0:25]
dados2 = iris.data[50:75]
dados3 = iris.data[100:125]
dados_treinamento = np.vstack([dados1, dados2, dados3])
#iris.target

#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(dados_treinamento)
kmeans.labels_
kmeans.cluster_centers_

#Prevendo novos valores
'Separando dados para teste'
dados4 = iris.data[25:50]
dados5 = iris.data[75:100]
dados6 = iris.data[125:150]
dados_teste = np.vstack([dados4, dados5, dados6])
c = kmeans.predict(dados_teste)
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
#iris.target

#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(dados_treinamento)
kmeans.labels_
kmeans.cluster_centers_

#Prevendo novos valores
'Separando dados para teste'
dados4 = wine.data[30:59]
dados5 = wine.data[94:130]
dados6 = wine.data[154:178]
dados_teste = np.vstack([dados4, dados5, dados6])
c = kmeans.predict(dados_teste)
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

from sklearn import metrics

'Separando dados para cálculo de desempenho'
dados7 = wine.target[30:59]
dados8 = wine.target[94:130]
dados9 = wine.target[154:178]
labels_true = np.hstack([dados7, dados8, dados9])

a = metrics.adjusted_rand_score(labels_true, c)


        


    


