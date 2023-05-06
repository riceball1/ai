"""
The following code is used to give a starting point to understand machine learning, that is part of the foundations for more complex AI models. This code is a simple linear regression model that can predict the output for a given input based on the trained parameters.

"""

import random

# Define the hypothesis function (linear regression)
def hypothesis(x, slope, intercept):
  return slope * x + intercept

# Define the loss function (mean squared error)
def loss(y, y_pred):
  return sum([(y[i][0] - y_pred[i]) ** 2 for i in range(len(y))]) / len(y)

# Define the gradient descent function
def gradient_descent(X, y, alpha=0.01, iterations=100):
  # Initialize the slope and intercept to random values
  slope = random.random()
  intercept = random.random()

  # Initialize the y_pred list with zeros
  y_pred = [0] * len(y)

  # Perform gradient descent
  for i in range(iterations):
    # Calculate the predicted y values
    y_pred = [hypothesis(X[j], slope, intercept) for j in range(len(X))]

    # Calculate the derivatives of the loss function with respect to slope and intercept
    d_slope = (-2 / len(X)) * sum([X[j] * (y[j][0] - y_pred[j]) for j in range(len(X))])
    d_intercept = (-2 / len(X)) * sum([y[j][0] - y_pred[j] for j in range(len(X))])

    # Update the slope and intercept
    slope -= alpha * d_slope
    intercept -= alpha * d_intercept

    # Print the loss every 10 iterations
    if (i+1) % 10 == 0:
      print("Iteration", i+1, "Loss:", loss(y, y_pred))

  # Return the final slope and intercept
  return slope, intercept

# Define the training data
X = [1, 2, 3, 4, 5]
y = [[3], [5], [7], [9], [11]]

# Train the model using gradient descent
slope, intercept = gradient_descent(X, y, alpha=0.01, iterations=100)

# Print the final slope and intercept
print("Final slope:", slope)
print("Final intercept:", intercept)
