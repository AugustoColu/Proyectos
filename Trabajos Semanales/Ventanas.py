import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift
import scipy.signal as sp

def eje_temporal (N, fs):
    
    Ts = 1/fs # t_final siempre va a ser 1 / df
    t_final = N * Ts # su inversa es la resolución espectral
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    return tt

def func_senoidal (tt, amp, frec, fase, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    return xx


N = 100
fs = 500
df = fs / N

tt = eje_temporal (N, fs)
nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral

x = func_senoidal (tt = tt, amp = 1, frec = 25, fase = 0, v_medio = 0)

w_1 = sp.windows.blackmanharris (N)
W_1 = fft (w_1, 2048) # esto calcula la FFT de w_1 con 2048 puntos de Zero-Padding
X = fft (x*w_1, 2048)
# observar que con solo W_1 = fft(w_1) no veo un carajo
eje = np.linspace (0, fs, len(W_1)) # defino los límites que quiero y reparto el intervalo en len(W_1) partes iguales

W_1dB = 20*np.log10(np.abs(fftshift(W_1))) # fftshift simplemente reordena los puntos de la FFT para centrarla en el gráfico
X = 20*np.log10(np.abs(fftshift(X)))

plt.subplot (2, 1, 1)
plt.plot (ff, x, color='black')
plt.plot (ff, w_1, color='orange')
plt.title ("Ventana en tiempo")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (eje, X, color='black')
plt.plot (eje, W_1dB, color='orange')
plt.title ("Ventana en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.grid ()

plt.tight_layout ()
plt.show ()