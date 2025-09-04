import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp

def eje_temporal (N, fs):
    
    Ts = 1/fs # t_final siempre va a ser 1 / df
    t_final = N * Ts # su inversa es la resolución espectral
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    return tt

def func_senoidal (tt, amp, frec, fase, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx


N = 1000
fs = 1000
df = fs / N # Resolución espectral

tt = eje_temporal (N, fs)
nn = np.arange (N) # Vector adimensional de muestras
ff = np.arange (N) * df # Vector en frecuencia al escalar las muestras por la resolución espectral

x = func_senoidal (tt = tt, amp = 1, frec = (N/4)*df, fase = 0, v_medio = 0) 
# observar que (N/4)*df = (N/4)*(fs/N) = fs/4, por ende no importa la cantidad de muestras, siempre la frecuencia será N/4
X = fft (x)

xx = func_senoidal (tt = tt, amp = 1, frec = ((N/4)+0.5)*df, fase = 0, v_medio = 0)
XX = fft (xx)


### Gráficos de señales ###

plt.subplot (3, 1, 1)
#plt.plot (nn, x, color='black', label='125 Hz') # Eje de abscisas adimensional (cantidad de muestras)
#plt.plot (tt, x, color='black', label='250 Hz') # Eje de abscisas en tiempo
plt.plot (ff, x, color='black', label='125 Hz') # Eje de absicas en frecuencia
plt.plot (ff, xx, color='green', label='250 Hz')
plt.title ("Señales")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("Amplitud")
plt.xlim (0, N/2) # ...visualizo hasta Nyquist
plt.legend ()
plt.grid (True)

### Gráficos de FFT's ###

plt.subplot (3, 1, 2)
#plt.plot (ff, np.abs(X), color='black') # Eje de ordenadas adimensional (N/2)
plt.plot (ff, np.log10(np.abs(X))*10, color='black') # Eje de ordenadas adimensional en dB
plt.plot (ff, np.log10(np.abs(XX))*10, marker='x', color='green')
plt.title ("Módulo de la señal en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, N/2)
plt.legend ()
plt.grid (True)

### Gráficos de fases ###

plt.subplot (3, 1, 3)
plt.plot (ff, np.angle(X), color='black')
plt.plot (ff, np.angle(XX), color='green')
plt.title ("Fase de la señal en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("Angulo")
plt.xlim (0, N/2)
plt.legend ()
plt.grid (True)

plt.tight_layout ()
plt.show ()

# %% Parseval

x_norm = (x - np.mean(x)) / (np.var(x))**(1/2)
# x_norm = func_senoidal (tt=tt, amp=np.sqrt(2), frec=100, fase=0, v_medio=0)
print ("Varianza =", np.var(x_norm), " ->  SD =", np.std(x_norm), " ->  Media =", np.mean(x_norm))

### Verifico Parseval ###

A = np.sum ((np.abs(x))**2)
B = np.sum ((np.abs(X))**2) / N
print (A-B)

# %% Zero-Padding

zeros = np.zeros (len(x)*9)
xPadding = np.concatenate ((x, zeros))
XPadding = fft (xPadding)

#ttPadding = eje_temporal (N = 10*N, fs = 1000)
ttPadding = np.arange (10*N) * (fs / (10*N))

plt.plot (ttPadding, np.log10(np.abs(XPadding)*10), linestyle='', marker='x')
plt.plot (ff, np.log10(np.abs(X)*10))
# plt.xlim (0, 500)
plt.grid (True)
plt.show ()
