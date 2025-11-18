# Day 1 Code Reading
from sklearn.datasets import load_iris
data = load_iris()

print("Features shape:", data.data.shape)
print("Classes:", data.target_names)