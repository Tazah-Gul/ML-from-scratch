import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
def accuracy(y_true, y_pred):
    return np.sum(y_true==y_pred)/len(y_true)
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Classification Accuracy is: ", accuracy(y_test, predictions))

