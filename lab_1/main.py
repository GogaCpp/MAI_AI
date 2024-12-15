import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class GradientDescent:
    def __init__(self, gradient_func, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
        self.gradient_func = gradient_func
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.history = [] # для визуализации

    def optimize(self, initial_point):
        x = np.array(initial_point)
        for i in range(self.max_iterations):
            grad = self.gradient_func(x)
            x_new = x - self.learning_rate * grad
            self.history.append(x.copy()) #сохраняем для визуализации

            if np.linalg.norm(x_new - x) < self.tolerance:
                break
            x = x_new
        return x

    def plot_results(self, func, analytical_solution):
        x_values = np.linspace(-5, 5, 100)
        y_values = func(x_values)

        plt.plot(x_values, y_values, label='Функция')
        plt.plot(analytical_solution[0],func(analytical_solution[0]), 'go', markersize=8, label='Аналитическое решение')

        history_array = np.array(self.history)
        plt.plot(history_array[:,0], func(history_array[:,0]), 'ro-', markersize=4, label='Найденное решение')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.title('Градиентный спуск')
        plt.show()

    def animate_results(self, func, analytical_solution):
        fig, ax = plt.subplots()
        x_values = np.linspace(-5, 5, 100)
        y_values = func(x_values)
        line, = ax.plot(x_values, y_values, label='Функция')
        point, = ax.plot([], [], 'ro', markersize=8)
        ax.plot(analytical_solution[0],func(analytical_solution[0]), 'go', markersize=8, label='Аналитическое решение')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.set_title('Градиентный спуск (Анимация)')


        def animate(i):
            point.set_data(self.history[i][0], func(self.history[i][0]))
            return point,


        ani = FuncAnimation(fig, animate, frames=len(self.history), interval=50, blit=True)
        plt.show()



# Пример использования
def f(x):
    return x**2 + 2*x + 1

def grad_f(x):
    return 2*x + 2


gd = GradientDescent(grad_f)
solution = gd.optimize([2])
print(f"Найденный минимум: {solution}")
analytical_solution = (-1,0)
gd.plot_results(f, analytical_solution)
gd.animate_results(f, analytical_solution)



#Пайплайн тестирования (пример):
results = []
for i in range(10):
    gd = GradientDescent(grad_f, learning_rate=0.1, initial_point = [i])
    solution = gd.optimize([i])
    error = abs(solution[0] - analytical_solution[0])
    results.append(error)

print(f"Погрешности для 10 запусков: {results}")
