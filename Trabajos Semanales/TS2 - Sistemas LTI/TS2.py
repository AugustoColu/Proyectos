import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp




y = np.zeros (len(x))

for n in range (0, N, Ts):
    y[n] = 0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2] + 1.5*y[n-1] - 0.5*y[n-2]
    