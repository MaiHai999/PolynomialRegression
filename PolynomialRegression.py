import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self , degree = 3):
        self.degree = degree

    def predict(self,X):
        x1 = self.create_polynomial_feature_set(X)
        return np.dot(x1, self.w) + self.b

    def lostFuction(self,y_true, y_hat):
        loss = np.mean((y_hat - y_true) ** 2)
        return loss

    def update_wight(self, X,y_true, y_pred):
        num_rows = X.shape[0]
        dw = (1 / num_rows) * np.dot(X.T, (y_pred - y_true))
        db = (1 / num_rows) * np.sum((y_pred - y_true))
        return dw, db

    def create_polynomial_feature_set(self , X):
        listx = []
        for i in X:
            a = np.power(i, np.arange(1, self.degree + 1))
            listx.append(a)
        return np.array(listx)

    def train(self, X, y, epochs, lr , batch_size):
        x = self.create_polynomial_feature_set(X)
        m, n = x.shape
        self.w = np.zeros((n, 1))
        self.b = 0
        y = y.reshape(m, 1)
        for epoch in range(epochs):
            for i in range((m - 1) // batch_size + 1):
                # Defining batches.
                start_i = i * batch_size
                end_i = start_i + batch_size
                x_batch = x[start_i:end_i]
                y_batch = y[start_i:end_i]

                # Calculating hypothesis
                y_hat = np.dot(x_batch, self.w) + self.b

                # Getting the gradients of loss w.r.t parameters.
                dw, db = self.update_wight(x_batch, y_batch, y_hat)

                # Updating the parameters.
                self.w -= lr * dw
                self.b -= lr * db

            loss = self.lostFuction(y, np.dot(x, self.w) + self.b)
            print(loss)








