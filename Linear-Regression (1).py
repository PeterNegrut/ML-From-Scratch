import sympy as sp
import random
import matplotlib.pyplot as plt


def generate_data(data_points):
    x = []
    y = []
    for i in range(data_points):
        x.append(random.randint(-10, 10))
        y.append(random.randint(-10, 10))
    return x, y

def function(m, b, data_points):
    x = []
    y = []
    noise = 0
    
    for i in range(data_points):
        noise = random.uniform(-2, 2)
        x_i = random.randint(-10, 10)
        y_i = m*x_i + b + noise
        x.append(x_i)
        y.append(y_i)
    return x, y

def display_data( x, y, m, b):

    plt.axline((0, b), slope = m, color = 'red', label = 'line')
    plt.plot(x, y, 'o')
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.show()

def MSE(m, b, x, y):
    s_sum = 0
    for i in range(len(x)):
        s_sum += ((y[i]) - (m*x[i] + b))**2
    return s_sum/len(x)

def regression(m, b, x, y, alpha):
    sum_m_resid=0
    sum_b_resid=0
    for i in range(len(x)):
        r = y[i] - (m*x[i] + b)
        sum_m_resid += r*x[i]
        sum_b_resid += r
    
    new_m = m - alpha*(-2/len(x) * sum_m_resid)
    new_b = b - alpha*(-2/len(x) * sum_b_resid)

    return new_m, new_b

def main():
    prev_MSE = float("inf")
    final_m = 0
    final_b = 0
    counter = 0
    m = random.randint(1, 10)
    b = random.randint(1, 10)

    x, y = function(m, b, 100)
    max_iterations = 20

    for i in range(max_iterations):
        counter+=1
        m, b = regression(m,b,x,y, 0.01)
        current = MSE(m, b, x, y)
        
        if (abs(current - prev_MSE) < 0.0001):
            break

        if (prev_MSE < current):
            final_m, final_b = m, b
        prev_MSE = current
        print(f"Step: {counter}, error: {current}")


    if (final_m and final_b != 0):
        print(f"Final m: {final_m}, Final b: {final_b}")
    display_data(x, y, m, b)
    


if __name__ == "__main__":
    main()