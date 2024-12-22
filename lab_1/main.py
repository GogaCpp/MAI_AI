import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from function import beale, beale_gradient, rastrigin, rastrigin_gradient, numerical_gradient
from GradientDescent import adam, gradient_descent, momentum_gradient_descent

def test_optimization(func, grad_func, initial_points, learning_rate, tolerance, max_iterations, num_runs=5, method=gradient_descent, **method_kwargs):
    results = []
    for initial_point in initial_points:
        x_opt, y_opt, x_hist, y_hist, f_hist = method(func, grad_func, initial_point, learning_rate, tolerance, max_iterations, **method_kwargs)
        results.append({'x': x_opt, 'y': y_opt, 'history': (x_hist, y_hist, f_hist)})

        # Визуализация
        fig, ax = plt.subplots()
        X = np.arange(-5, 5, 0.1)
        Y = np.arange(-5, 5, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = func(X, Y)
        contour = ax.contour(X, Y, Z, 20)
        point, = ax.plot([], [], 'ro')
        def animate(i):
            point.set_data(x_hist[:i+1], y_hist[:i+1])
            return point,
        ani = animation.FuncAnimation(fig, animate, frames=len(x_hist), interval=100)
        plt.title(f'Градиентный спуск. Запуск {initial_point}')
        plt.show()
    return results


def calculate_error(results, x_analytical, y_analytical):
    errors = []
    for result in results:
        x_opt = result['x']
        y_opt = result['y']
        error = np.sqrt((x_opt - x_analytical)*2 + (y_opt - y_analytical)*2)
        errors.append(error)
    return errors


initial_points = [(-0.1, 0.1)]
learning_rate = 0.01
tolerance = 1e-6
max_iterations = 1000
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# results_rastrigin = test_optimization(rastrigin, rastrigin_gradient, initial_points, learning_rate, tolerance, max_iterations, method=gradient_descent)

# results_rastrigin_momentum = test_optimization(
#     rastrigin, rastrigin_gradient, initial_points,
#     learning_rate, tolerance, max_iterations,
#     method=adam, momentum=momentum, beta1=beta1,
#     beta2=beta2, epsilon=epsilon
#     )

# #results_rosenbrock = test_optimization(rosenbrock, rosenbrock_gradient, initial_points, learning_rate, tolerance, max_iterations) # аналогично для Розенброка

# errors = calculate_error(results_rastrigin_momentum, 0, 0)
# print(errors)

results_beale = test_optimization(beale, beale_gradient, initial_points, learning_rate, tolerance, max_iterations, method=gradient_descent)

# results_beale_momentum = test_optimization(
#     beale, beale_gradient, initial_points,
#     learning_rate, tolerance, max_iterations,
#     method=adam, momentum=momentum, beta1=beta1,
#     beta2=beta2, epsilon=epsilon
#     )

# #results_rosenbrock = test_optimization(rosenbrock, rosenbrock_gradient, initial_points, learning_rate, tolerance, max_iterations) # аналогично для Розенброка

errors = calculate_error(results_beale, 0, 0)
print(errors)