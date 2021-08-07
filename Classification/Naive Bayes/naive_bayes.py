import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_claases = len(self._classes)
        # initialize mean, variance, priors
        self.mean = np.zeros((n_claases, n_features), dtype=np.float64)
        self.var = np.zeros((n_claases, n_features), dtype=np.float64)
        self.priors = np.zeros(n_claases, dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self.mean[c,:] = X_c.mean(axis=0)
            self.var[c,:] = X_c.var(axis=0)
            self.priors [c]= X_c.shape[0] / float(n_samples)




    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._probability_density_func(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _probability_density_func(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator