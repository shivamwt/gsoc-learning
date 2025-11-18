"# Day 2 Code Reading" 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeClassifier()

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
print("Model accuracy:", accuracy_score(y_test, pred))
