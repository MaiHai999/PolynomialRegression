# Polynomial Regression

This project demonstrates how to implement Polynomial Regression from scratch using Python and NumPy. It provides a simple and clear understanding of how polynomial regression works by breaking down the process without relying on machine learning libraries.

## Features

- **Custom Polynomial Regression Implementation**: No machine learning libraries like scikit-learn are used; the algorithm is implemented using NumPy.
- **Data Fitting**: Fit a polynomial to a given set of data points.
- **Prediction**: Predict new values based on the fitted polynomial model.
- **Visualization**: Plot the data points and the fitted polynomial curve for better understanding.

## Technologies Used

- **Python**: The core programming language.
- **NumPy**: For numerical computations, including matrix operations and polynomial fitting.
- **Matplotlib**: For plotting data points and the polynomial curve.

## Project Structure

- `polynomial_regression.py`: Contains the implementation of the polynomial regression algorithm.
- `data.csv`: Sample dataset used for fitting the polynomial.
- `plot.py`: Script for visualizing the data points and the fitted polynomial curve.

## How It Works

1. **Load Data**: Load the dataset containing the independent variable (X) and dependent variable (Y).
2. **Feature Transformation**: Transform the independent variable into polynomial features.
3. **Fit the Model**: Use the normal equation to compute the coefficients of the polynomial.
4. **Predict**: Use the computed coefficients to make predictions for new data.
5. **Visualize**: Plot the original data points along with the polynomial curve.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/polynomial-regression.git
   cd polynomial-regression
