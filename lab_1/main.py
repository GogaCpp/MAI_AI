from GradientDescent import GradientDescent
import numpy as np
import Ploting


def rastrigin(x):
    A = 10
    return A * 2 + (x[0]**2 + x[1]**2) - A * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))


def rastrigin_grad(x):
    A = 10
    return np.array([2 * x[0] + A * 2 * np.pi * np.sin(2 * np.pi * x[0]),
                     2 * x[1] + A * 2 * np.pi * np.sin(2 * np.pi * x[1])])


def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_grad(x):
    a = 1
    b = 100
    return np.array([-2 * (a - x[0]) - 2 * b * (x[1] - x[0]**2) * x[0],
                     2 * b * (x[1] - x[0]**2)])


def test_gradient_descent(function, gradient):
    initial_point = [3.0, 3.0]
    gd = GradientDescent(function, gradient, learning_rate=0.1)
    optimal_point, history = gd.optimize(initial_point)

    print("Найденная точка оптимума:", optimal_point)
    print("Значение функции в оптимуме:", function(optimal_point))

    # Вычисление погрешности
    analytical_optimum = [0, 0]
    error = np.linalg.norm(np.array(optimal_point) - np.array(analytical_optimum))
    print("Погрешность:", error)

    # Визуализация
    Ploting.visualize_optimization(function, history)


test_gradient_descent(rastrigin, rastrigin_grad)
