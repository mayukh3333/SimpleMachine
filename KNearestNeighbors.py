import numpy as np

class KNearestNeighbors():
    """

    """

    def __init__(self, X_train, Y_train, K = 3):
        """

        """

        self.X_train      = X_train
        self.Y_train      = Y_train
        self.K            = K
        self.neighbours   = []
        self.distance     = []
        self.predictions  = []


    @staticmethod
    def eucledian_dist(v1,v2):
        """


        """
        return np.sqrt(np.sum((v1-v2)**2,axis = 1))

    def fit(self, X_test, returndist = False):
        """

        """

        neighbours1 = []
        distance1   = []
        dist  = [self.eucledian_dist(self.X_train,test) for test in X_test]
        for row in dist:
            d = enumerate(row)
            s = sorted(d, key=lambda x: x[1])[:self.K]
            nearest_neig = [tup[0] for tup in s]
            nearest_dist = [tup[1] for tup in s]
            neighbours1.append(nearest_neig)
            distance1.append(nearest_dist)

        self.neighbours = neighbours1
        self.distance   = distance1

        if returndist:

            return neighbours1,distance1

        else:

            return neighbours1

    def predict(self, X_test):
        """


        """
        y_pred = []

        for i in self.neighbours:
            r1 = [self.Y_train[r2] for r2 in i]
            y_pred.append(np.argmax(np.bincount(r1)))

        self.predictions = y_pred

        return y_pred

    def score(self, X_test, y_test):
        """

        """
        y_pred = self.predict(X_test)

        return float(sum(y_pred == y_test)) / float(len(y_test))
