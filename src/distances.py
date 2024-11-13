import numpy as np


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    mat = np.empty([X.shape[0],Y.shape[0]])

    for i in range(X.shape[0]):
        feature1 = X[i,:]
        for j in range(Y.shape[0]):
            distance = feature1 - Y[j,:]
            dist_sq = np.power(distance,2)
            sum_sq = np.sum(dist_sq)
            mat[i,j] = np.sqrt(sum_sq)
           
            

    return mat





def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """

    matrix = np.empty([X.shape[0],Y.shape[0]])

    for i in range(X.shape[0]):
        feature1 = X[i,:]
        for j in range(Y.shape[0]):
            distance = np.absolute(feature1 - Y[j,:])
            matrix[i,j] = np.sum(distance)

    return matrix



def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """


    def magnitude(vector):
        return np.sqrt(np.sum(np.power(vector, 2)))

    matrix = np.empty([X.shape[0],Y.shape[0]])

    for i in range(X.shape[0]):
        feature1 = X[i,:]
        for j in range(Y.shape[0]):
            numerator = np.dot(feature1, Y[j,:])
            mag_u = magnitude(feature1)
            mag_v = magnitude(Y[j,:])
            matrix[i,j] = 1- (numerator/(mag_u*mag_v))

    return matrix
