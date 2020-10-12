
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import load_wine
from sklearn import metrics

#Carrega o wine dataset em wine
wine = load_wine()

#Separando dados para treinamento
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

#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=11,weights="uniform", p=1)
neigh.fit(X, y)

#Separando dados para teste
dados4 = wine.data[30:59]
dados5 = wine.data[94:130]
dados6 = wine.data[154:178]
dados_teste = np.vstack([dados4, dados5, dados6])
#Prevendo novos valores
print(neigh.predict(dados_teste))
c = neigh.predict(dados_teste)

#Analise de desempenho
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

print(f"N iteracoes: ")

print(f"Sum das distancias: {neigh.effective_metric_}")
print(f"% de acertos: {indice}")


#Score de similaridade
dados7 = wine.target[30:59]
dados8 = wine.target[94:130]
dados9 = wine.target[154:178]
labels_true = np.hstack([dados7, dados9, dados8])
print(labels_true)
print(c)
a = metrics.adjusted_rand_score(labels_true, c)
print(f"score de similaridade: {a}")
