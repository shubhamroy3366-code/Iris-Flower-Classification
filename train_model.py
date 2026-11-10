
# Step 1: Import libraries

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Step 2: Load flower dataset

iris = load_iris()

X = iris.data
y = iris.target


# Step 3: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 4: Create AI model

model = RandomForestClassifier(random_state=42)


# Step 5: Train AI model

model.fit(X_train, y_train)


# Step 6: Make predictions

predictions = model.predict(X_test)


# Step 7: Check accuracy

accuracy = accuracy_score(y_test, predictions)

print("Flower Classification Project")
print("Model Accuracy:", accuracy)


# Step 8: Show flower names

print("Flower names:", iris.target_names)

# Step 9: Save trained model

import pickle

with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("AI Model Saved Successfully!")
