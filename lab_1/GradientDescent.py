import numpy as np


class GradientDescent:
    def __init__(self, function, gradient, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        self.function = function
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    def optimize(self, initial_point):
        point = np.array(initial_point)
        history = [point.copy()]

        for i in range(self.max_iter):
            grad = self.gradient(point)
            new_point = point - self.learning_rate * grad

            history.append(new_point.copy())

            if np.linalg.norm(new_point - point) < self.tolerance:
                break
            point = new_point

        return point, history
