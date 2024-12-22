import pygmo as pg
import numpy as np
import pandas as pd


# Определение функции Растригина
class Rastrigin:
    def __init__(self):
        self.dim = 2  # Размерность
        self.lower_bounds = [-5.12] * self.dim
        self.upper_bounds = [5.12] * self.dim

    def fitness(self, x):
        A = 10
        return [A * self.dim + sum((xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x))]

    def get_bounds(self):
        return (self.lower_bounds, self.upper_bounds)


# Определение функции Била
class Rosenbrock:
    def __init__(self):
        self.dim = 2  # Размерность (можно изменить на большее)
        self.lower_bounds = [-5] * self.dim
        self.upper_bounds = [5] * self.dim

    def fitness(self, x):
        return [(1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2]

    def get_bounds(self):
        return (self.lower_bounds, self.upper_bounds)


# Создание экземпляров задач
rastrigin_problem = pg.problem(Rastrigin())
rosenbrock_problem = pg.problem(Rosenbrock())

# Алгоритмы для тестирования
algorithms = {
    'Simulated Annealing': pg.algorithm(pg.simulated_annealing()),
    'Genetic Algorithm': pg.algorithm(pg.sga()),
    'CMA-ES': pg.algorithm(pg.cmaes())
}

# Сохраняем результаты
results = []

for name, algo in algorithms.items():
    # Оптимизация функции Растригина
    pop_rastrigin = pg.population(rastrigin_problem, size=10)  # Размер популяции
    pop_rastrigin = algo.evolve(pop_rastrigin)
    rastrigin_optimum = pop_rastrigin.champion_x
    rastrigin_value = pop_rastrigin.champion_f[0]

    # Оптимизация функции Била
    pop_rosenbrock = pg.population(rosenbrock_problem, size=10)  # Размер популяции
    pop_rosenbrock = algo.evolve(pop_rosenbrock)
    rosenbrock_optimum = pop_rosenbrock.champion_x
    rosenbrock_value = pop_rosenbrock.champion_f[0]

    # Сохраняем результаты
    results.append({
        'Algorithm': name,
        'Rastrigin Optimum': rastrigin_optimum,
        'Rastrigin Value': rastrigin_value,
        'Rosenbrock Optimum': rosenbrock_optimum,
        'Rosenbrock Value': rosenbrock_value,
    })
# Настройка отображения всех строк и столбцов
pd.set_option('display.max_rows', None)  # None означает неограниченное количество строк
pd.set_option('display.max_columns', None)  # None означает неограниченное количество столбцов
pd.set_option('display.expand_frame_repr', False)  # Не разбивать длинные строки
# Создание таблицы с результатами
results_df = pd.DataFrame(results)
results_df.to_csv("lab_2/test.csv")  # Сохраняю так как в уонсоли не удобно
print(results_df)