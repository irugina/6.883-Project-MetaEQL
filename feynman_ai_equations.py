import numpy as np
import random


equation_dict = {
    "id": lambda x: x, #this is not from feynman ai
    "gaussian1": lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi),
    "gaussian2": lambda x, sigma: np.exp(-(x/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma),
    "gaussian": lambda x, x_0, sigma: np.exp(-((x-x_0)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma),
    "euclidean": lambda x1, x2, y1, y2: np.sqrt((x2-x1)**2+(y2-y1)**2),
    "sin": lambda x: np.sin(10 * x - 1),
    "exp": lambda x: np.exp(x),
    "f1": lambda x: np.sin(x) * x - 3,
    "f2": lambda x: x**2 + 3*x + 1,
}


def get_equation_kinematic():
    case = random.randint(0, 3)
    func = None
    if case == 0:
        # No acceleration
        m = random.uniform(0.5, 2)

        def func(x):
            return m * x

    elif case == 1:
        a = random.uniform(-1, 1)
        b = random.uniform(-1, 1)
        c = random.uniform(-1, 1)

        def func(x):
            return a * x**2 + b * x + c

    return func


def get_equation_wave():
    case = random.randint(0, 1)
    func = None
    if case == 0:
        # Sinusoidal
        f = random.uniform(0.5, 2)
        phi = random.uniform(0, 2*np.pi)

        def func(x):
            return np.sin(f*2*np.pi*x + phi)

    elif case == 1:
        # Sum of exponentials
        a = random.uniform(-1, 1)
        b = random.uniform(-1, 1)

        def func(x):
            return a * np.exp(x) + b * np.exp(-x)

    return func
