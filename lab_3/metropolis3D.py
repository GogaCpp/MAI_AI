import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Определяем функцию плотности для двумерного нормального распределения
def target_pdf_3d(x, y):
    pos = np.dstack((x, y))  # Объединяем x и y в одну матрицу
    rv = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    return rv.pdf(pos)  # Вычисляем плотность вероятности

# Метод Метрополиса-Гастинга в 2D
def metropolis_hastings_2d(pdf, initial, iterations):
    current = initial
    samples = [current]
    for _ in range(iterations):
        proposal = current + np.random.normal(0, 0.5, size=2)  # Предложение нового значения в 2D
        acceptance_ratio = pdf(proposal[0], proposal[1]) / pdf(current[0], current[1])
        if np.random.rand() < acceptance_ratio:
            current = proposal
        samples.append(current)
    return np.array(samples)

# Генерация точек
initial_2d = np.array([0.0, 0.0])  # Начальная точка
iterations = 1000
mh_samples_2d = metropolis_hastings_2d(target_pdf_3d, initial_2d, iterations)

# Визуализация блуждания в 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Подготовка данных для 3D поверхности
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = target_pdf_3d(X, Y)  # Используем X и Y для получения плотностей

# Отображаем 3D поверхность
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

# Отобразим точки, полученные методом Метрополиса-Гастинга
ax.scatter(mh_samples_2d[:, 0], mh_samples_2d[:, 1], target_pdf_3d(mh_samples_2d[:, 0], mh_samples_2d[:, 1]), color='red', alpha=0.5, s=5, label='Metropolis-Hastings Samples')

ax.set_title('3D Metropolis-Hastings Sampling')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Density')
ax.legend()
plt.show()