git init
git add .
git commit -m "Initial commit: logistic regression from scratch"
git remote add origin https://github.com/Tank4676/logistic_regression_model_scratch-.git
git branch -M main
git push -u origin main
git add README.md
git commit -m "Add partial README"
git push

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris.data", header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Convert to binary classification: Iris-setosa = 1, others = 0
df['label'] = (df['class'] == 'Iris-setosa').astype(int)

# Features and labels
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['label'].values.reshape(-1, 1)

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add bias term (intercept)
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss
def compute_loss(y, y_pred):
    m = y.shape[0]
    return -1/m * np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

# Gradient Descent function
def logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    losses = []

    for _ in range(epochs):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        dw = 1/m * np.dot(X.T, (y_pred - y))
        weights -= lr * dw

    return weights, losses

# Train the model
weights, losses = logistic_regression(X, y)

# Prediction function
def predict(X, weights, threshold=0.5):
    probs = sigmoid(np.dot(X, weights))
    return (probs >= threshold).astype(int)

# Make predictions
y_pred = predict(X, weights)

# Evaluation without sklearn
accuracy = np.mean(y_pred == y)
TP = np.sum((y_pred == 1) & (y == 1))
FP = np.sum((y_pred == 1) & (y == 0))
FN = np.sum((y_pred == 0) & (y == 1))
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

# Print results
print("Final Weights:", weights.flatten())
print("Final Loss:", losses[-1])
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Plot loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.show()
