import numpy as np

def gradient_descent(func, grad_func, initial_point, learning_rate, tolerance, max_iterations):
    x, y = initial_point
    x_history = [x]
    y_history = [y]
    f_history = [func(x, y)]

    for i in range(max_iterations):
        grad_x, grad_y = grad_func(x, y)
        x_new = x - learning_rate * grad_x
        y_new = y - learning_rate * grad_y

        if np.sqrt((x_new - x)**2 + (y_new - y)**2) < tolerance:
            break

        x, y = x_new, y_new
        x_history.append(x)
        y_history.append(y)
        f_history.append(func(x, y))

    return x, y, x_history, y_history, f_history


def momentum_gradient_descent(func, grad_func, initial_point, learning_rate, tolerance, max_iterations,**kwarg):
    x, y = initial_point
    vx, vy = 0, 0 # Скорости
    x_history = [x]
    y_history = [y]
    f_history = [func(x, y)]
    momentum = kwarg["momentum"]

    for i in range(max_iterations):
        grad_x, grad_y = grad_func(x, y)
        vx = momentum * vx - learning_rate * grad_x
        vy = momentum * vy - learning_rate * grad_y
        x += vx
        y += vy

        if np.sqrt((vx)**2 + (vy)**2) < tolerance: #Проверка остановки по скорости
            break

        x_history.append(x)
        y_history.append(y)
        f_history.append(func(x, y))

    return x, y, x_history, y_history, f_history

def adam(func, grad_func, initial_point, learning_rate, tolerance, max_iterations, **kwarg):
    x, y = initial_point
    m_x, m_y = 0, 0 # Моменты первого порядка
    v_x, v_y = 0, 0 # Моменты второго порядка
    x_history = [x]
    y_history = [y]
    f_history = [func(x, y)]
    beta1 = kwarg["beta1"]
    beta2 = kwarg["beta2"]
    epsilon = kwarg["epsilon"]

    for i in range(max_iterations):
        grad_x, grad_y = grad_func(x, y)
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        v_x = beta2 * v_x + (1 - beta2) * grad_x**2
        v_y = beta2 * v_y + (1 - beta2) * grad_y**2

        m_x_hat = m_x / (1 - beta1**(i + 1))
        m_y_hat = m_y / (1 - beta1**(i + 1))
        v_x_hat = v_x / (1 - beta2**(i + 1))
        v_y_hat = v_y / (1 - beta2**(i + 1))

        x_new = x - learning_rate * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
        y_new = y - learning_rate * m_y_hat / (np.sqrt(v_y_hat) + epsilon)

        if np.sqrt((x_new - x)**2 + (y_new - y)**2) < tolerance:
            break

        x, y = x_new, y_new
        x_history.append(x)
        y_history.append(y)
        f_history.append(func(x, y))

    return x, y, x_history, y_history, f_history


