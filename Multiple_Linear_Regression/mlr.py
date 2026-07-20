import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Target Variable
y_name = input("Enter Target Variable Name: ")
y = np.array(list(map(float, input(f"Enter {y_name} values (comma separated): ").split(","))))

# Number of Features
n = int(input("Enter Number of Features: "))

X = []
feature_names = []

for i in range(n):
    name = input(f"Enter Feature {i+1} Name: ")
    feature_names.append(name)

    

    values = list(map(float, input(f"Enter {name} values: ").split(",")))

    if len(values) != len(y):
        print("Error: Number of X and Y values must be equal.")
        exit()

    X.append(values)

# Convert to proper format
X = np.array(X).T

# Train Model
model = LinearRegression()
model.fit(X, y)

# Prediction on Training Data
y_pred = model.predict(X)

# Equation
print("\n===== Multiple Linear Regression =====")
print(f"Intercept = {model.intercept_:.4f}")

equation = f"{y_name} = {model.intercept_:.4f}"

for i in range(n):
    print(f"Coefficient of {feature_names[i]} = {model.coef_[i]:.4f}")
    equation += f" + ({model.coef_[i]:.4f} × {feature_names[i]})"

print("\nEquation:")
print(equation)

# Predict New Data
choice = input("\nDo you want prediction? (yes/no): ").lower()

if choice == "yes":
    sample = []

    for i in range(n):
        sample.append(float(input(f"Enter {feature_names[i]}: ")))

    result = model.predict([sample])

    print(f"\nPredicted {y_name} = {result[0]:.4f}")

# Metrics
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n===== Performance =====")
print("MSE :", round(mse,4))
print("MAE :", round(mae,4))
print("RMSE:", round(rmse,4))
print("R²  :", round(r2,4))

# Actual vs Predicted Graph
plt.figure(figsize=(6,4))
plt.scatter(y, y_pred, color="blue")
plt.plot([min(y), max(y)], [min(y), max(y)], color="red")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

# Performance Graph
metrics = ["R2", "RMSE", "MAE", "MSE"]
values = [r2, rmse, mae, mse]

plt.figure(figsize=(6,4))
plt.scatter(metrics, values)
plt.grid(True)
plt.title("Performance Metrics")

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

plt.show()