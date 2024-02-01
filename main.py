import matplotlib.pyplot as plt
from PolynomialRegression import PolynomialRegression
import numpy as np



np.random.seed(42)

X = np.random.rand(1000, 1)
y = 5 * ((X) ** (9)) + np.random.rand(1000, 1)
X1 = np.random.rand(500, 1)

poly = PolynomialRegression(9)
poly.train(X, y, 20, 0.001 , 5)
y_hat = poly.predict(X1)

# Plotting
fig = plt.figure(figsize=(8, 6))
plt.plot(X, y, 'y.')
plt.plot(X1, y_hat, 'r.')
plt.legend(["True Data Points", "Predictions from Polynomial Regression"])
plt.xlabel('X - Input')
plt.ylabel('y - target / true')
plt.title('Polynomial Regression')
plt.show()
