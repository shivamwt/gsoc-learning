# Day 3 Visualization
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()

X = data.data
y = data.target
target_names = data.target_names

# Scatter plot of feature 1 vs feature 2
plt.figure(figsize=(8, 6))

for target in [0, 1, 2]:
    plt.scatter(
        X[y == target, 0],
        X[y == target, 1],
        label=target_names[target]
    )

plt.xlabel("Feature 1 (Sepal Length)")
plt.ylabel("Feature 2 (Sepal Width)")
plt.legend()
plt.title("Iris – Visualization of First Two Features")

plt.show()

