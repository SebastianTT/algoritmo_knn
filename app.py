from flask import Flask
import feedparser
import csv
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)
cont =0
@app.route("/knn")
def main():
	url_test = 'test.csv'
	url_train = 'train.csv'
	training_data = []
	test_data = []
	print("datos totales")
	print(test_data)
	print("----------------------------------")
	df_test = pd.read_csv(url_test)
	df_train = pd.read_csv(url_train)
	print('\nEstadísticas del dataset:')
	print(df_train.describe())
	print(df_test.describe())
	print('\nMedia')
	print(df_test.mean())
	print(df_train.mean())
	print("\nPrecisión")
	X = np.array(df_train.drop(['species'], 1))
	y = np.array(df_train['species'])
	print("train",np.array(df_train['species']))
	print(X)
	print(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	knn = KNeighborsClassifier(n_neighbors = 3)
	knn.fit(X_train, y_train)
	Y_pred = knn.predict(X_test)
	prediccion_knn = knn.predict(df_test)
	print("Aqui esta el df_test", df_test.iloc[:,:-1].values)
	print("prediccion del KNN")
	print(prediccion_knn)
	print('Precisión Vecinos más Cercanos:')
	print(knn.score(X_train, y_train))
	print("Matriz de confusion")
	print(confusion_matrix(y_test, Y_pred))
	print("Reporte de clasificacion")
	print(classification_report(y_test, Y_pred))
	jst = {str(i):prediccion_knn[i] for i in range(len(prediccion_knn))}
	jst["saludo"] ="hola"
	return jst #dict({"informacion_knn":[{i:prediccion_knn[i] for i in range(len(prediccion_knn))},{"hola":"holin"}]})
if __name__ == "__main__":
    app.run(debug=True)