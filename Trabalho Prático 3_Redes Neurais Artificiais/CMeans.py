from sklearn.cluster import KMeans
import numpy as np

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

#Ajustar dados de test
dados1 = iris.data[0:25]
dados2 = iris.data[50:75]
dados3 = iris.data[100:125]
dados_treinamento = np.vstack([dados1, dados2, dados3])


# scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# we fit the train data
scaler.fit(dados_treinamento)

# scaling the train data
train_data = scaler.transform(dados_treinamento)

dados4 = iris.data[25:50]
dados5 = iris.data[75:100]
dados6 = iris.data[125:150]
dados_teste = np.vstack([dados4, dados5, dados6])
test_data = scaler.transform(dados_teste)


kmeans = KMeans(n_clusters=3, random_state=1).fit(train_data)
kmeans.labels_
kmeans.cluster_centers_
predictions_test = kmeans.predict(test_data)

dados4 = iris.target[25:50]
dados5 = iris.target[75:100]
dados6 = iris.target[125:150]
labels_true = np.hstack([dados4, dados5, dados6])
print(labels_true)
pd.crosstab(labels_true,predictions_test, rownames=['Real'], colnames=['          Predito'], margins=True)
predictions_test[predictions_test == 0] = 5
predictions_test[predictions_test == 1] = 3
predictions_test[predictions_test == 2] = 4
predictions_test[predictions_test == 4] = 1
predictions_test[predictions_test == 5] = 2
predictions_test[predictions_test == 3] = 0
print(predictions_test)

from sklearn.metrics import classification_report
print(classification_report(predictions_test, labels_true))
