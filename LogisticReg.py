import numpy as np

class LogisticRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):

        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        #init parameters
        n_samples, n_fea = X.shape
        self.weights = np.zeros(n_fea)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iters):
            linear = np.dot(X,self.weights)+self.bias
            y_pred = self.sigmoid(linear)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))

            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        linear = np.dot(X, self.weights)+self.bias
        y_pred = self.sigmoid(linear)
        return [1 if i > 0.5 else 0 for i in y_pred]

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return list(map(lambda x,y: x == y, y_true,y_pred)).count(True)/len(y_true)
