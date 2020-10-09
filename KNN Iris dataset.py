# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
''' 
#Carrega o iris dataset em iris 
iris = load_iris()
'Separando dados para treinamento'
dados1 = iris.data[0:25]
dados2 = iris.data[50:75]
dados3 = iris.data[100:125]
dados_treinamento = np.vstack([dados1, dados2, dados3])
X = dados_treinamento
dados_target1 = iris.target[0:25]
dados_target2 = iris.target[50:75]
dados_target3 = iris.target[100:125]
dados_target = np.hstack([dados_target1, dados_target2, dados_target3])
y = dados_target 
#iris.target
#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=5,weights="uniform")
neigh.fit(X, y)

'Separando dados para teste'
dados4 = iris.data[25:50]
dados5 = iris.data[75:100]
dados6 = iris.data[125:150]
dados_teste = np.vstack([dados4, dados5, dados6])
#Prevendo novos valores
print(neigh.predict(dados_teste))
c = neigh.predict(dados_teste)

cont = 0
for i in range(25):
    if c[i] == 0:
        cont = cont + 1
        
for i in range(25,50):
    if c[i] == 1:
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
X = dados_treinamento

dados_target1 = wine.target[0:30]
dados_target2 = wine.target[59:94]
dados_target3 = wine.target[130:154]
dados_target = np.hstack([dados_target1, dados_target2, dados_target3])
y = dados_target 
#iris.target
#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=5,weights="uniform")
neigh.fit(X, y)

'Separando dados para teste'
dados4 = wine.data[30:59]
dados5 = wine.data[94:130]
dados6 = wine.data[154:178]
dados_teste = np.vstack([dados4, dados5, dados6])
#Prevendo novos valores
print(neigh.predict(dados_teste))
c = neigh.predict(dados_teste)

cont = 0
for i in range(29):
    if c[i] == 0:
        cont = cont + 1
        
for i in range(29,65):
    if c[i] == 1:
        cont = cont + 1
        
for i in range(65,89):
    if c[i] == 2:
        cont = cont + 1

'índice de desempenho'
indice = (cont/c.size)*100
print(indice)
