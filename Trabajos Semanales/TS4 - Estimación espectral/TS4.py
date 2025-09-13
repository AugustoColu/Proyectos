import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp


def eje_temporal (N, fs):

    Ts = 1/fs
    t_final = N * Ts
    tt = np.arange (0, t_final, Ts)

    return tt


def gen_señal (tt, df, amp, frec, SNR, R, fase = 0, v_medio = 0):

    N = len(tt)
    frec_rand = frec + np.random.uniform (-2, 2) * df
    s = np.tile (amp * np.sin (2*np.pi * frec_rand * tt + fase) + v_medio, reps=(1, R))

    if SNR == None:
      return s

    pot_ruido = amp_0**2 / (2*10**(SNR/10))
    ruido = np.tile (np.random.normal (0, np.sqrt(pot_ruido), N), reps=(R, 1))
    # print (f"Potencia de ruido -> {np.var(ruido):3.3f}") # verifico 

    return s + ruido # modelo de señal

"""
¿Cómo sé si calculé bien la FFT para la matriz de senoidales?
Matriz de senoidales: X = (N x R)
... si el eje de simería del vector está en R/2, está como el orto
... si el eje de simetría está en N/2, tamos bien, millonardo
"""

R = 200 # realizaciones
SNR = 10 # SNR en dB
amp_0 = np.sqrt(2) # amplitud [V]
N = 1000 # cantidad de muestras
fs = 1000 # frecuencia de muestreo [Hz = 1/seg]
df = fs / N # resolución espectral [Hz]
frec = (N/4)*df

nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral
tt = eje_temporal (N = N, fs = fs)

x_1 = gen_señal (tt = tt, df = df, amp = amp_0, frec = frec, SNR = None, R = R)
X_1 = (1/N) * fft(x_1, axis=0)


plt.plot (10*np.log10(2*np.abs(X_1)**2))
# plt.grid (True)
# plt.legend ()
# plt.show ()