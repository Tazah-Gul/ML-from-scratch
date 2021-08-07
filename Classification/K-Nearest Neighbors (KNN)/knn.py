import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distance
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        # find k nearest samples and their labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)    # 1 is for very first most common
        return most_common[0][0]
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x1-x2)))