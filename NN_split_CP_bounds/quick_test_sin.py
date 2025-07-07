# Required packages
import numpy as np
import matplotlib.pyplot as plt

# Pinball loss function
def pinball_loss(q, s, alpha):
    if q >= s:
        return alpha * (q - s)
    else:
        return (1 - alpha) * (s - q)

# Generate synthetic data with two noise regions
np.random.seed(42)
x1 = np.linspace(0, 5, 50)  # Region 1: Low noise
x2 = np.linspace(5, 10, 50)  # Region 2: High noise
x = np.concatenate((x1, x2))

# True signal (sin wave)
y_true1 = np.sin(x1) + 0.05 * np.random.randn(len(x1))  # Low noise region
y_true2 = np.sin(x2) + 0.5 * np.random.randn(len(x2))   # High noise region
y_true = np.concatenate((y_true1, y_true2))

# Predicted quantile with alpha = 0.9
alpha = 0.9  # Upper quantile
predicted_quantile = np.sin(x) + alpha * 0.3  # Predicted values

# Calculate pinball loss for alpha = 0.9
pinball_loss_values = [pinball_loss(pred, true, alpha) for pred, true in zip(predicted_quantile, y_true)]

# Calculate the cumulative pinball loss differences to detect group boundaries
cumulative_loss = np.cumsum(pinball_loss_values)

# Detect the group boundary as the point with the largest change in cumulative loss
boundary_index = np.argmax(np.abs(np.diff(cumulative_loss)))
boundary_x = x[boundary_index]

# Plotting the true function and predicted quantile
plt.figure(figsize=(12, 6))

# Plot the true signal
plt.plot(x, y_true, label='True Signal', color='black', linestyle='-', alpha=0.8)

# Plot the predicted quantile
plt.plot(x, predicted_quantile, label=f'Predicted Quantile (alpha={alpha})', linestyle='--')
plt.fill_between(x, y_true, predicted_quantile, alpha=0.3, label=f'Pinball Loss (alpha={alpha})')

# Automatically mark the calculated group boundary
plt.axvline(x=boundary_x, color='red', linestyle=':', label=f'Calculated Group Boundary (x={boundary_x:.2f})')

plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title(f'Group Separation using Pinball Loss (Alpha={alpha})')
plt.legend()
plt.grid(True)
plt.show()
