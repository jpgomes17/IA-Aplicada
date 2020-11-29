from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

 #Carrega o iris dataset em iris
iris = load_iris()
'Separando dados para treinamento'
dados1 = iris.data[0:25]
dados2 = iris.data[50:75]
dados3 = iris.data[100:125]
dados_treinamento = np.vstack([dados1, dados2, dados3])

dados_target1 = iris.target[0:25]
dados_target2 = iris.target[50:75]
dados_target3 = iris.target[100:125]
dados_target = np.hstack([dados_target1, dados_target2, dados_target3])


#print(model.output_dict["variable"][0])
neigh = KNeighborsClassifier(n_neighbors=7 ,weights="uniform", p=1)
neigh.fit(dados_treinamento, dados_target)

dados4 = iris.data[25:50]
dados5 = iris.data[75:100]
dados6 = iris.data[125:150]
dados_teste = np.vstack([dados4, dados5, dados6])

dados4 = iris.target[25:50]
dados5 = iris.target[75:100]
dados6 = iris.target[125:150]
test_target = np.hstack([dados4, dados5, dados6])

predictions_test = neigh.predict(dados_teste)
print(predictions_test)
print(test_target)

from sklearn.metrics import classification_report
print(classification_report(predictions_test, test_target))
