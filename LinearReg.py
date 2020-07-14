import numpy as np

class LinearRegression:

    def __init__(self, alpha = 0.01, iterations = 2000):

        self.alpha = alpha
        self.iterations = iterations
        self.X = []
        self.y = []
        self.theta = np.zeros((2,1))
        self.costs = []

    def cost_function(self, X, y):

        m = len(y)
        y_pred = X.dot(self.theta)
        error = (y_pred - y)**2

        return 1/(2 *m) * np.sum(error)

    def fit(self, X, y):

        m = len(y)
        self.X = np.append(np.ones((m,1)),X.reshape(m,1),axis =1)
        self.y = y.reshape(m,1)
        for i in range(self.iterations):
            y_pred = self.X.dot(self.theta)
            error = np.dot(self.X.transpose(), (y_pred - self.y))
            self.theta -= self.alpha * 1/m * error
            self.costs.append(self.cost_function(self.X, self.y))

    def predict(self, x):

        self.theta.squeeze()

        return self.theta[0] + self.theta[1]*x
