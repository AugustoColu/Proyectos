import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

def func_senoidal (amp, frec, fase, tt, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx


N = 60
tt = np.arange (0, N-1, 1) / N
x = func_senoidal (10, 1, 0, tt, 0)

X = np.zeros (N, dtype = np.complex128)

for k in range (0, N-1, 1):
    for n in range (0, N-1, 1):
        X[k] += x[n] * np.exp(-1j*k*2*np.pi*n/N)
    print (X[k])
    
    
# Pasaje de N (adimensional) a frecuencia -> tt * Δf

        
plt.subplot (2, 1, 1)
plt.plot (tt, x, linestyle='', marker='s', color='black')
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (X, color='green')
plt.grid (True)

plt.show ()
        
    