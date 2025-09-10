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

x = func_senoidal (tt = tt, amp = 1, frec = 10, fase = 0, v_medio = 0)


### Ventana tipo BlackmanHarris ###

w_1 = sp.windows.blackmanharris (N)
W_1 = fft (w_1, 2048)
# observar que con solo W_1 = fft(w_1) no veo un carajo
X_1 = fft (x*w_1, 2048) # esto calcula la FFT de x con 2048 puntos de Zero-Padding

eje = np.linspace (0, fs, len(W_1)) # defino los límites que quiero y reparto el intervalo en len(W_1) partes iguales

W_1dB = 20*np.log10(np.abs(fftshift(W_1))) # fftshift simplemente reordena los puntos de la FFT para centrarla en el gráfico
X_1dB = 20*np.log10(np.abs(fftshift(X_1)))

plt.figure (1)

plt.subplot (2, 1, 1)
plt.plot (ff, x, color='black', label='x')
plt.plot (ff, w_1, color='orange', label='w_1')
plt.title ("BlackmanHarris en tiempo")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("Amplitud [V]")
plt.legend ()
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (eje, X_1dB, color='black', label='X_1dB')
plt.plot (eje, W_1dB, color='orange', label='W_1dB')
plt.title ("Ventana en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.legend ()
plt.grid (True)

plt.tight_layout ()


### Ventana tipo FlatTop ###

w_2 = sp.windows.flattop (N)
W_2 = fft (w_2, 2048)
X_2 = fft (x*w_2, 2048) # esto calcula la FFT de x con 2048 puntos de Zero-Padding

W_2dB = 20*np.log10(np.abs(fftshift(W_2)))
X_2dB = 20*np.log10(np.abs(fftshift(X_2)))

plt.figure (2)

plt.subplot (2, 1, 1)
plt.plot (ff, x, color='black', label='x')
plt.plot (ff, w_2, color='orange', label='w_2')
plt.title ("FlatTop en tiempo")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("Amplitud [V]")
plt.legend ()
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (eje, X_2dB, color='black', label='x_2dB')
plt.plot (eje, W_2dB, color='orange', label='W_2dB')
plt.title ("Ventana en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.legend ()
plt.grid (True)

plt.tight_layout ()


### Ventana tipo Hamming ###

w_3 = sp.windows.hamming (N)
W_3 = fft (w_3, 2048)
X_3 = fft (x*w_3, 2048) # esto calcula la FFT de x con 2048 puntos de Zero-Padding

W_3dB = 20*np.log10(np.abs(fftshift(W_3)))
X_3dB = 20*np.log10(np.abs(fftshift(X_3)))

plt.figure (3)

plt.subplot (2, 1, 1)
plt.plot (ff, x, color='black', label='x')
plt.plot (ff, w_3, color='orange', label='w_3')
plt.title ("Hamming en tiempo")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("Amplitud [V]")
plt.legend ()
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (eje, X_3dB, color='black', label='x_3dB')
plt.plot (eje, W_3dB, color='orange', label='W_3dB')
plt.title ("Ventana en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.legend ()
plt.grid (True)

plt.tight_layout ()

plt.show ()
