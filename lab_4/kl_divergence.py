import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, multivariate_normal

# Генерация примера данных для второго набора
# Например, данные, полученные методом Гиббса
gibbs_samples = np.random.normal(loc=0, scale=1, size=(1000, 2))

# Функция для оценки плотности и вычисления KL-дивергенции
def calculate_kl_divergence(samples_1, samples_2):
    # Оценка плотности для обеих выборок
    kde_1 = gaussian_kde(samples_1.T)
    kde_2 = gaussian_kde(samples_2.T)

    # Создание сетки для оценки плотности
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    # Вычисляем плотности
    p_1 = kde_1(grid_points.T)
    p_2 = kde_2(grid_points.T)

    # Нормализуем плотности
    p_1 /= np.sum(p_1)
    p_2 /= np.sum(p_2)

    # Избегаем деления на ноль
    p_1 += 1e-10
    p_2 += 1e-10

    # Вычисляем KL-дивергенцию
    kl_divergence_1_to_2 = np.sum(p_1 * np.log(p_1 / p_2))
    kl_divergence_2_to_1 = np.sum(p_2 * np.log(p_2 / p_1))

    return kl_divergence_1_to_2, kl_divergence_2_to_1

# Получение KL-дивергенции для выборок
# kl_mh_to_gibbs, kl_gibbs_to_mh = calculate_kl_divergence(mh_samples_2d, gibbs_samples)

# print(f"KL Divergence from Metropolis-Hastings to Gibbs: {kl_mh_to_gibbs}")
# print(f"KL Divergence from Gibbs to Metropolis-Hastings: {kl_gibbs_to_mh}")