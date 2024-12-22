import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def visualize_optimization(function, history):
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])

    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z, levels=50)
    ax.clabel(contour, inline=True, fontsize=8)

    # Анимация
    points = np.array(history)

    scat = ax.scatter(points[:, 0], points[:, 1], color='red')

    def update(frame):
        scat.set_offsets(points[frame])
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(points), interval=200, blit=True)
    plt.title('Градиентный спуск')
    plt.show()