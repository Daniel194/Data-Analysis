import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

colors = 10 * ["g", "r", "c", "b", "k"]

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()


class K_means:
    def __init__(self, k=2, tolerance=0.001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self, data):
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

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = prev_centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))

        return classification


clf = K_means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.clasifications:
    color = colors[classification]

    for featureset in clf.clasifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

plt.show()
