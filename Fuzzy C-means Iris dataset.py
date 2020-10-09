# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_iris
import numpy as np

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

'''
#Carrega o iris dataset em iris 
iris = load_iris()
'Separando dados para treinamento'
dados1 = iris.data[0:25]
dados2 = iris.data[50:75]
dados3 = iris.data[100:125]
dados_treinamento = np.vstack([dados1, dados2, dados3])
alldata = dados_treinamento.transpose()
label = iris.target 
ncenters = 3
#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)



# Generate uniformly sampled data spread across the range [0, 10] in x and y
'Separando dados para teste'
dados4 = iris.data[25:50]
dados5 = iris.data[75:100]
dados6 = iris.data[125:150]
dados_teste = np.vstack([dados4, dados5, dados6])
newdata = dados_teste

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)
cluster_membership_predict = np.argmax(u, axis=0)
c = cluster_membership_predict
print(c)

cont = 0
for i in range(25):
    if c[i] == 2:
        cont = cont + 1
        
for i in range(25,50):
    if c[i] == 1:
        cont = cont + 1
        
for i in range(50,75):
    if c[i] == 0:
        cont = cont + 1

'índice de desempenho'
indice = (cont/c.size)*100
print(indice)
'''



from sklearn.datasets import load_wine

#Carrega o wine dataset em wine 
wine = load_wine()
'Separando dados para treinamento'
dados1 = wine.data[0:30]
dados2 = wine.data[59:94]
dados3 = wine.data[130:154]
dados_treinamento = np.vstack([dados1, dados2, dados3])
alldata = dados_treinamento.transpose()
label = wine.target 
ncenters = 3
#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)



# Generate uniformly sampled data spread across the range [0, 10] in x and y
'Separando dados para teste'
dados4 = wine.data[30:59]
dados5 = wine.data[94:130]
dados6 = wine.data[154:178]
dados_teste = np.vstack([dados4, dados5, dados6])
newdata = dados_teste

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)
cluster_membership_predict = np.argmax(u, axis=0)
c = cluster_membership_predict
print(c)

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
print(indice)
