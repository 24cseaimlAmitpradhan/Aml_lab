import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 4])


m, c = np.polyfit(x, y, 1)

y_pred = m * x + c


mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)

ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)



print("\n===== Linear Regression Result =====")

print(f"Slope (m)      : {m:.2f}")
print(f"Intercept (c)  : {c:.2f}")
print(f"Equation       : y = {m:.2f}x + {c:.2f}")

print("\n===== Evaluation Metrics =====")

print(f"MAE   : {mae:.2f}")
print(f"MSE   : {mse:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.2f}")

plt.figure(figsize=(5,4))
plt.scatter(x, y, color="blue")
plt.plot(x, y_pred, color="red")
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()


metrics = ["MAE", "MSE", "RMSE", "R²"]
values = [mae, mse, rmse, r2]

plt.figure(figsize=(5,4))

plt.scatter(metrics, values, s=100)

plt.title("Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.grid(True)

for i in range(len(metrics)):
    plt.text(metrics[i], values[i] + 0.02, round(values[i], 2),
             ha='center')

plt.show()
