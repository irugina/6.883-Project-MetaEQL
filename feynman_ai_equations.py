import numpy as np
import random


equation_dict = {
    "id": lambda x: x, # this is not from feynman ai
    "gaussian1": lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi),
    "gaussian2": lambda x, sigma: np.exp(-(x/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma),
    "gaussian": lambda x, x_0, sigma: np.exp(-((x-x_0)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma),
    "euclidean": lambda x1, x2, y1, y2: np.sqrt((x2-x1)**2+(y2-y1)**2),
    "sin": lambda x: np.sin(10 * x - 1),
    "exp": lambda x: np.exp(x),
    "f1": lambda x: np.sin(x) * x - 3,
    "f2": lambda x: x**2 + 3*x + 1,

    'wave1': lambda x: np.sin(2*np.pi*x),
    'wave2': lambda x: np.sin(np.pi*x + np.pi/2),
    'wave3': lambda x: 0.5 * np.exp(-x) * np.sin(0.5*np.pi*x + 0.7*np.pi/2),
    'wave4': lambda x: np.sin(1.4*np.pi*x + 0.1),
    'wave5': lambda x: 0.7 * np.exp(-x),
    'wave6': lambda x: -0.4 * np.exp(-x),
    'wave7': lambda x: 0.3 * np.exp(-x) * np.sin(np.pi*x),
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

        name = "case_{}_p1_{}_p2_{}".format(case, f, phi)

    elif case == 1:
        # Sum of exponentials
        a = random.uniform(-1, 1)
        b = random.uniform(-1, 1)

        def func(x):
            return a * np.exp(x) + b * np.exp(-x)

        name = "case_{}_p1_{}_p2_{}".format(case, a, b)

    return func, name

def wave_exp(number_train, number_val):
    equation_dict = dict()
    func_names, val_func_names = [], []
    for _ in range(number_train):
        func, name = get_equation_wave()
        equation_dict[name] = func
        func_names.append(name)
    for _ in range(number_val):
        func, name = get_equation_wave()
        equation_dict[name] = func
        val_func_names.append(name)
    return func_names, val_func_names, equation_dict

