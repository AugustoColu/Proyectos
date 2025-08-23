import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

def func_senoidal (amp, frec, fase, tt, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx


N = 8
eje_N = np.linspace (0, N-1, N)
x = np.zeros (N)
x[4] = 1 # creo una delta(x-4)

X = np.zeros (N, dtype = np.complex128)

for k in range (0, N, 1):
    for n in range (0, N, 1):
        X[k] += x[n] * np.exp(-1j*k*2*np.pi*n/N)
    print (X[k])
    
    
# Pasaje de N (adimensional) a frecuencia -> tt * Δf

plt.subplot (2, 1, 1)
plt.stem (eje_N, x)
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (eje_N, X, linestyle='-', marker='o', color='black')
plt.grid (True)

plt.show ()