import numpy as np
import matplotlib.pyplot as plt

# Input
x = np.array(list(map(float, input("Enter X values (comma separated): ").split(","))))
y = np.array(list(map(float, input("Enter Y values (comma separated): ").split(","))))

# Check if lengths are equal
if len(x) != len(y):
    print("Error: Number of X values and Y values must be equal.")
    exit()

# Linear Regression
m, c = np.polyfit(x, y, 1)

# Predicted Values
y_pred = m * x + c

# Evaluation Metrics
mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)

ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

# Print Results
print("\n===== Linear Regression Result =====")
print(f"Slope (m)      : {m:.2f}")
print(f"Intercept (c)  : {c:.2f}")
print(f"Equation       : y = {m:.2f}x + {c:.2f}")

print("\n===== Evaluation Metrics =====")
print(f"MAE   : {mae:.2f}")
print(f"MSE   : {mse:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.2f}")

# Sort data for plotting regression line
index = np.argsort(x)
x_sorted = x[index]
y_pred_sorted = y_pred[index]

# Regression Graph
plt.figure(figsize=(6,4))
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x_sorted, y_pred_sorted, color="red", label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Evaluation Metrics Graph (Points Only)
metrics = ["MAE", "MSE", "RMSE", "R²"]
values = [mae, mse, rmse, r2]

plt.figure(figsize=(6,4))
plt.scatter(metrics, values, s=100, color="green")

for i in range(len(metrics)):
    plt.text(metrics[i], values[i], f"{values[i]:.2f}",
             ha="center", va="bottom")

plt.title("Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.grid(True)
plt.show()