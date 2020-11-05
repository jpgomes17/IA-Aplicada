# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""
import random
import collections
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from geneticalgorithm import geneticalgorithm as ga

#função para conferir a resposta do algoritimo abixo com os mesmos dados de entrada
def ft(n):
    
     # Implementa o Algoritmo KNN
    neigh = KNeighborsClassifier(n_neighbors=n, weights="uniform")
    neigh.fit(x_treino, y_treino)
    # Prevendo novos valores

    d = neigh.predict(x_test)

    erro = 0
    contador = 0

    # CONTADOR PARA AVALIAR GRAU DE ACERTO
    for c in range(len(y_test)):
        if y_test[c] == d[c]:
            contador = contador + 1
        else:
            erro = erro + 1
    
    return  print(erro,d)


def f(X):
 
    # Implementa o Algoritmo KNN
    neigh = KNeighborsClassifier(n_neighbors=int(X), weights="uniform")
    neigh.fit(x_treino, y_treino)
    # Prevendo novos valores

    d = neigh.predict(x_test)

    erro = 0
    contador = 0

    # CONTADOR PARA AVALIAR GRAU DE ACERTO
    for c in range(len(y_test)):
        if y_test[c] == d[c]:
            contador = contador + 1
        else:
            erro = erro + 1
            
    #lista.append(int(X))
    #lista1.append(erro)
    #d=collections.Counter(lista)
    #print(d)
    #print(collections.Counter(lista1))
  
    
    return -(1-(erro/len(y_test)))*100


#Carrega o iris dataset em iris
iris = load_iris()


#variaveis
lista=[]
lista1=[]
dado_randomicos=[]
x_treino=[]
y_treino=[]
x_test=[]
y_test=[]
# armazenar o predict


lista_comparacao=[ x for x in range(len(iris.data))]
percentual=0.80
#GERANDO NUMEROS RANDOMICOS NÃO REPETIDOS, PARA COLETAR OS DADOS DE FORMA MAIS IMPARCIAL POSSIVEL
while 1:
    aleatorio=random.randrange(0,len(iris.target))
    if aleatorio not in dado_randomicos:
        dado_randomicos.append(aleatorio)
    if round(len(iris.target)*percentual)==len(dado_randomicos):
        break
dado_randomicos=np.array(dado_randomicos)
#ADICIONANDO DADOS NAS LISTAS
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
x_treino=np.array(x_treino)
y_treino=np.array(y_treino)

varbound=np.array([[1,50]])
model=ga(function=f,dimension=1,variable_type='int',variable_boundaries=varbound)
model.run()
erro = model.best_function















