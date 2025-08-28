import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp
from scipy.io import wavfile

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


amplitud = 0
frec = 0
fase = 0
N = 1000
fs = 1000
v_medio = 0


tt = eje_temporal (N, fs)
ss = func_senoidal (1, N/4, 0, tt, 0)

plt.subplot (3, 1, 1)
plt.plot (tt, ss, linestyle='-', color='black')
plt.title ("Señal sinusoidal de 1 KHz")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

fft_ss = fft (ss)

plt.subplot (3, 1, 2)
plt.plot (tt, np.abs(fft_ss), linestyle='-', color='green')
#plt.stem (np.abs(fft_ss))
plt.title ("Módulo de la señal en frecuencia")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

plt.subplot (3, 1, 3)
plt.plot (tt, np.angle(fft_ss), linestyle='-', color='green')
plt.title ("Fase de la señal en frecuencia")
plt.xlabel ("Tiempo")
plt.ylabel ("Angulo")
plt.grid (True)

plt.tight_layout ()
plt.show ()

