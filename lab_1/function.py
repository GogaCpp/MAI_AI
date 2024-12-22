import numpy as np

def rastrigin(x, y):
    return 10 * 2 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

def rastrigin_gradient(x, y):
    grad_x = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    grad_y = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
    return grad_x, grad_y

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (x - 1)**2

def rosenbrock_gradient(x, y):
    grad_x = -400 * x * (y - x**2) + 2 * (x - 1)
    grad_y = 200 * (y - x**2)
    return grad_x, grad_y

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def beale_gradient(x, y):
    grad_x = 2*(1.5 - x + x*y)*(y-1) + 2*(2.25 - x + x*y**2)*(y**2 - 1) + 2*(2.625 - x + x*y**3)*(y**3 -1)
    grad_y = 2*(1.5 - x + x*y)*x + 2*(2.25 - x + x*y**2)*2*x*y + 2*(2.625 - x + x*y**3)*3*x*y**2
    return grad_x, grad_y

def numerical_gradient(func, x, y, h=1e-6):
    grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h)
    grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return grad_x, grad_y

def numerical_gradient_rosenbrock(func, x, y, h=1e-6):
    grad_x = (rosenbrock(x + h, y) - rosenbrock(x - h, y)) / (2 * h)
    grad_y = (rosenbrock(x, y + h) - rosenbrock(x, y - h)) / (2 * h)
    return grad_x, grad_y