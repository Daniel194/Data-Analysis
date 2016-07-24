import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

colors = 10 * ["g.", "r.", "c.", "b.", "k."]

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()


class K_means:
    def __init__(self, k=2, tolerance=0.001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

    def predict(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.clasifications = {}

            for i in range(self.k):
                self.clasifications[i] = []

            for featureset in X:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.clasifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.clasifications:
                self.centroids[classification] = np.average(self.clasifications[classification], axis=0)

    def fit(self, data):
        pass
