import numpy as np

class LogisticRegressionFromScratch():
    def __init__(self, learning_rate=0.0001, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    # Train the Model
    def fit(self, X, y):

        # m => Number of rows
        # n => Number of features
        self.m, self.n = X.shape


        # Initializing weights and bias as zeros
        self.w = np.zeros(self.n)
        self.b = 0

        # Setting features and Labels to class
        self.X = X
        self.y = y


    # Training using Gradient Descent
        for _ in range(self.n_epochs):
            self.update_weights()


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    # Update weights using Gradient Descent
    def update_weights(self):

        # y_pred using sigmoid function
        y_pred = self.sigmoid(self.X.dot(self.w) + self.b)


        # Partial Derivaties of sum of Binary Cross Entropy Loss Function or Log Loss Function
        dw = (1/self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (1/self.m) * np.sum(y_pred - self.y)


        # updating the weights and bias
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db


    def predict(self, X):
        y_pred = self.sigmoid(X.dot(self.w) + self.b)
        y_pred = np.where(y_pred > 0.5, 1, 0)

        return y_pred
    
logisticRegressionFromScratchModel = LogisticRegressionFromScratch()

