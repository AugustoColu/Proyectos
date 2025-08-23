import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

def eje_temporal (N, fs):
    
    # Resolución espectral = fs / N
    # t_final siempre va a ser 1/Res. espec.
    Ts = 1/fs
    t_final = N * Ts # su inversa es la resolución espectral
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    return tt

def func_senoidal (amp, frec, fase, tt, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx

N = 8
fs = 50
frec = 10

vec_N = np.arange (N) # es lo mismo que np.arange (0, N, 1)
tt = eje_temporal (N, fs)

x = func_senoidal (3, frec, 0, tt, 4)

y = np.zeros (len(tt))

plt.plot (tt, x, marker='*')
plt.plot (tt, y)

plt.show ()
