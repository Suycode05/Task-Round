import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
a = iris.data 
b = iris.target

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(a_train, b_train)

b_pred = knn.predict(a_test)

acc = accuracy_score(b_test, b_pred)
print(f'Accuracy: {acc:.2f}')

print('Confusion Matrix:')
print(confusion_matrix(b_test, b_pred))

print('Classification Report:')
print(classification_report(b_test, b_pred))
