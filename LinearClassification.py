import numpy as np

# Linear Classifier (Perceptron)
class LinearClassifierFromScratch:
    def __init__(self, learning_rate=0.001, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs


    def fit(self, X, y):

        # m => Number of rows
        # n => Number of features
        self.m, self.n = X.shape

        # Initializing weights and bias as zeros
        self.w = np.zeros(self.n)
        self.b = 0

        for _ in range(self.n_epochs):
            linear_model = np.dot(X, self.w) + self.b

            # Calculating predictions using activation function
            predictions = self.activation_function(linear_model)

            # Calculating Error
            errors = y - predictions

            # Update weights and bias
            self.w += self.learning_rate * np.dot(errors, X)
            self.b += self.learning_rate * np.sum(errors)


    def activation_function(self, z):
        return np.where(z > 0, 1, 0)


    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        predictions = self.activation_function(linear_model)

        return predictions
    
linearClassifierFromScratchModel = LinearClassifierFromScratch()