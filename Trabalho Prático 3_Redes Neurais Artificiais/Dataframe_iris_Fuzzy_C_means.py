# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_iris
import numpy as np
#import matplotlib.pyplot as plt
import collections
import random

#Carrega o iris dataset em iris
iris = load_iris()

dado_randomicos=[]
x_treino=[]
y_treino=[]
x_test=[]
y_test=[]
# armazenar o predict
d=[]

lista_comparacao=[ x for x in range(len(iris.data))]
percentual=0.50

while 1:
    aleatorio=random.randrange(0,len(iris.target))
    if aleatorio not in dado_randomicos:
        dado_randomicos.append(aleatorio)
    if len(iris.target)*percentual==len(dado_randomicos):
        break

for n in dado_randomicos:
    x_treino.append(iris.data[n])
    y_treino.append(iris.target[n])

    if n in  lista_comparacao:
        lista_comparacao.remove(n)

for n in lista_comparacao:
    x_test.append(iris.data[n])
    y_test.append(iris.target[n])

x_test=np.array(x_test)
y_test=np.array(y_test)

alldata3=x_test.transpose()

x_treino=np.array(x_treino)
xcont=collections.Counter(y_treino)


# print(iris.data)
# print(iris.feature_names)
# print(list(iris.target_names))
alldata = x_treino
# np.vstack(iris.data[50:150,:])

alldata2 = alldata.transpose()
label = iris.target

# a=np.vstack((iris.data[:coleta1,:],iris.data[50:(50+coleta2),:],iris.data[100:(100+coleta3),:]))
# b=np.hstack((iris.target[:coleta1],iris.target[50:(50+coleta2)],iris.target[100:(100+coleta3)]))
# a1=np.vstack((iris.data[20:50,:],iris.data[50:100,:],iris.data[100:110,:]))
# b1=np.hstack((iris.target[20:50],iris.target[50:100],iris.target[100:110]))
# alldata=a
# alldata2=alldata.transpose()
# label=b

ncenters = 3
#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
alldata2, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)

# c=collections.Counter(Newcluster_membership[:50])
# c1=np.array(c.most_common())
# c=collections.Counter(Newcluster_membership[50:100])
# c2=np.array(c.most_common())
# c=collections.Counter(Newcluster_membership[100:150])
# c3=np.array(c.most_common())


# print('Classificação: ',c1[0,0],'\n',(c1[0,1]/50)*100)
# print('Classificação:',c2[0,0],'\n',(c2[0,1]/50)*100)
# print('Classificação:',c3[0,0],'\n',(c3[0,1]/50)*100)
# print(((c1[0,1]+c2[0,1]+c3[0,1])/150)*100)
# """

# Generate uniformly sampled data spread across the range [0, 10] in x and y
# newdata = escolher os dados a serem testados

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model



u1, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(alldata3
    , cntr, 2, error=0.005, maxiter=1000)
Newcluster_membership=np.argmax(u1,axis=0)
predictions_test = Newcluster_membership
print(Newcluster_membership)

c=collections.Counter(Newcluster_membership[:50-xcont[0]])
print(c)
c1=np.array(c.most_common())
c=collections.Counter(Newcluster_membership[50-xcont[0]:100-xcont[0]-xcont[1]])
print(c)
c2=np.array(c.most_common())
c=collections.Counter(Newcluster_membership[100-xcont[1]-xcont[0]:150-xcont[2]-xcont[1]-xcont[0]])
print(c)
c3=np.array(c.most_common())


print('Classificação: ',c1[0,0],'\n',(c1[0,1]/(50-xcont[0]))*100)
print('Classificação:',c2[0,0],'\n',(c2[0,1]/(50-xcont[1]))*100)
print('Classificação:',c3[0,0],'\n',(c3[0,1]/(50-xcont[2]))*100)
print(((c1[0,1]+c2[0,1]+c3[0,1])/(150-xcont[2]-xcont[1]-xcont[0]))*100)
# """

v_0 = c1[0,0]
v_1 = c2[0,0]
v_2 = c3[0,0]
print(v_0,v_1, v_2)

predictions_test[predictions_test == v_0] = 3
predictions_test[predictions_test == v_1] = 4
predictions_test[predictions_test == v_2] = 5
predictions_test[predictions_test == 3] = 0
predictions_test[predictions_test == 4] = 1
predictions_test[predictions_test == 5] = 2

print(y_test)
print(predictions_test)

from sklearn.metrics import classification_report
print(classification_report(predictions_test, y_test))
