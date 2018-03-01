from PIL import Image
from numpy import *


def pca(X):
    """
    Principal Component Analysis
    :param X: X, matrix with training data stored as flattende arrays in rows
    :return: projection matrix ( with important dimensions first), variance and mean.
    """

    # get dimensions
    num_data, dim = X.shape

    # centered data
    mean_X = X.mean(axis=0)

    X -= mean_X

    if dim >= num_data:
        # PCA - compacted trick used
        M = dot(X, X.T)  # covariance matrix
        e, EV = linalg.eigh(M)  # eigenvalues and eigenvectors
        tmp = dot(X.T, EV)  # this is the compact trick
        V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1]  # reverse since eigenvalues are in increasting order

        for i in range(V.shape[1]):
            V[:, i] /= S

    else:
        # PCA - SVD used
        U, S, V = linalg.svd(X)
        V = V[:num_data]  # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean

    return V, S, mean_X
