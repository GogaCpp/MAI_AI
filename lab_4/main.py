import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from kl_divergence import calculate_kl_divergence


# Генерация случайных данных
# np.random.seed(42)
data = np.concatenate(
    [
        np.random.normal(loc=-2, scale=0.5, size=100),
        np.random.normal(loc=2, scale=0.5, size=100)
    ]
)

# Визуализация исходных данных
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data Histogram')

# Метод ядерного сглаживания
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data[:, np.newaxis])
x_d = np.linspace(-5, 5, 1000)
log_dens = kde.score_samples(x_d[:, np.newaxis])
plt.fill(x_d, np.exp(log_dens), color='blue', alpha=0.5, label='KDE Density Estimation')

plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# 1.2 Алгоритм EM
# Применение EM-алгоритма
gmm = GaussianMixture(n_components=2, random_state=42).fit(data[:, np.newaxis])
x = np.linspace(-5, 5, 1000).reshape(-1, 1)
logprob = gmm.score_samples(x)
pdf = np.exp(logprob)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data Histogram')
plt.plot(x, pdf, color='red', label='EM Density Estimation')
plt.title('EM Algorithm for Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()


# 2. Генерация новых точек с методами Метрополиса-Гастинга и Гиббса
# 2.1 Метод Метрополиса-Гастинга
def metropolis_hastings(pdf, initial, iterations):
    current = initial
    samples = [current]
    for _ in range(iterations):
        proposal = np.random.normal(current, 0.5)  # Предложение нового значения
        acceptance_ratio = pdf(proposal) / pdf(current)
        if np.random.rand() < acceptance_ratio:
            current = proposal
        samples.append(current)
    return np.array(samples)


# Функция плотности для использования в алгоритме
def target_pdf(x):
    if np.isscalar(x):  # Если x - это скаляр
        return np.exp(kde.score_samples(np.array([[x]])))  # Превращаем в массив с формой (1, 1)
    else:
        return np.exp(kde.score_samples(x[:, np.newaxis]))  # Если x - это массив


# Генерация новых точек
mh_samples = metropolis_hastings(target_pdf, initial=0.0, iterations=1000)

# Визуализация
plt.figure(figsize=(10, 6))
plt.hist(mh_samples, bins=30, density=True, alpha=0.5, color='green', label='Metropolis-Hastings Samples')
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data Histogram')
plt.fill(x_d, np.exp(log_dens), color='blue', alpha=0.5, label='KDE Density Estimation')
plt.title('Metropolis-Hastings Samples')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()


# 2.2 Метод Гиббса
#  ! может стоит по разным файлам разнести
def gibbs_sampling(initial, iterations):
    samples = [initial]
    for _ in range(iterations):
        # Новый выбор x из первого распределения
        current_x = np.random.normal(
            loc=-2, scale=0.5
        ) if np.random.rand() < 0.5 else np.random.normal(loc=2, scale=0.5)
        current_y = np.random.normal(loc=current_x ** 2, scale=0.5)  # Предположим, что y зависит от x
        samples.append((current_x, current_y))
    return np.array(samples)


# Генерация новых точек
gibbs_samples = gibbs_sampling((0, 0), iterations=1000)

# Визуализация
plt.figure(figsize=(10, 6))
plt.hist([s[0] for s in gibbs_samples], bins=30, density=True, alpha=0.5, color='orange', label='Gibbs Samples')
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Data Histogram')
plt.fill(x_d, np.exp(log_dens), color='blue', alpha=0.5, label='KDE Density Estimation')
plt.title('Gibbs Samples')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# Визуализация всех трех наборов
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Original Data')
plt.hist(mh_samples, bins=30, density=True, alpha=0.5, color='green', label='Metropolis-Hastings Samples')
plt.hist([s[0] for s in gibbs_samples], bins=30, density=True, alpha=0.5, color='orange', label='Gibbs Samples')
plt.fill(x_d, np.exp(log_dens), color='blue', alpha=0.5, label='KDE Density Estimation')
plt.title('Comparison of Original Data and Samples')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# kl_mh_to_gibbs, kl_gibbs_to_mh = calculate_kl_divergence(
#     gibbs_samples,
#     mh_samples
#     )

# print(f"KL Divergence from Metropolis-Hastings to Gibbs: {kl_mh_to_gibbs}")
# print(f"KL Divergence from Gibbs to Metropolis-Hastings: {kl_gibbs_to_mh}")