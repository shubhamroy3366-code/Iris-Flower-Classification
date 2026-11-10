
import pickle

# Load trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

print("🌸 IRIS FLOWER PREDICTION 🌸")

# Take flower measurements
sl = float(input("Enter Sepal Length: "))
sw = float(input("Enter Sepal Width: "))
pl = float(input("Enter Petal Length: "))
pw = float(input("Enter Petal Width: "))

# Make prediction
data = [[sl, sw, pl, pw]]

prediction = model.predict(data)

flowers = ["Setosa", "Versicolor", "Virginica"]

print("\nPredicted Flower:", flowers[prediction[0]])

