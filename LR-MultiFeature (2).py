import sympy as sp
import random
import numpy as np
import matplotlib.pyplot as plt


def generate_data(weights, data_points, b):

    sqrft = np.random.randint(500, 5001, data_points)
    age = np.random.randint(1, 11, data_points)
    beds = np.random.randint(1, 6, data_points)
    noise = np.random.uniform(-2, 2, data_points)

    X = np.column_stack((sqrft, age, beds))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    y = X @weights + b + noise
    return X, y

def display_results(X, y, weights, b):
    """
    Visualize how well your model fits the data.
    
    Args:
        X: features (n_samples, 3)
        y: actual prices (n_samples,)
        weights: learned weights (3,)
        b: learned bias
    """
    # Compute predictions
    y_pred = X @ weights + b
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot predictions vs actual
    plt.scatter(y, y_pred, alpha=0.5, edgecolors='k')
    
    # Perfect prediction line (y = x)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect predictions')
    
    plt.xlabel('Actual Price', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.title('Predicted vs Actual Prices', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² or MSE as text
    mse = np.mean((y - y_pred) ** 2)
    plt.text(0.05, 0.95, f'MSE: {mse:.2f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def MSE(weights, X, y, b):

    return np.mean((y - (X @ weights + b)) ** 2)


def regression_features(weights, X, y, b, alpha):

    error = (X@weights + b) - y
    update_w1 = weights[0] - alpha*(2/X.shape[0])*(np.sum(X[:, 0] * error))
    update_w2 = weights[1] - alpha*(2/X.shape[0])*(np.sum(X[:, 1] * error))
    update_w3 = weights[2] - alpha*(2/X.shape[0])*(np.sum(X[:, 2] * error))
    update_b = b - alpha*(2/X.shape[0]) * (np.sum(error))
    weights = np.array([update_w1, update_w2, update_w3])
    return update_w1, update_w2, update_w3, update_b

def main():
    prev_MSE = float("inf")

    counter = 0
    alpha = 0.01
    real_weights = np.array([20.0, -2.0, 5.0])
    weights = np.zeros(3)
    b = 10
    data_points = 100

    X, y = generate_data(real_weights, data_points, b)
    max_iterations = 2000

    for i in range(max_iterations):
        counter+=1
        w1, w2, w3, b = regression_features(weights, X, y, b, alpha)
        weights = np.array([w1, w2, w3])
        current = MSE(weights, X, y, b)

        
        if (abs(current - prev_MSE) < 0.0001):
            break

        prev_MSE = current
        print(f"Step: {counter}, error: {current}")
    print(f"Final w1: {weights[0]}, Final w2: {weights[1]}, Final w3: {weights[2]}")
    display_results(X, y, weights, b)
    
    


if __name__ == "__main__":
    main()   