import numpy as np 
equation_dict = {
        "id": lambda x: x, #this is not from feynman ai
        "exp1": lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi),
        "exp2": lambda x, sigma: np.exp(-(x/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma),
        "exp3": lambda x, x_0, sigma: np.exp(-((x-x_0)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma),
        "euclidean": lambda x1, x2, y1, y2: np.sqrt((x2-x1)**2+(y2-y1)**2),
        }
