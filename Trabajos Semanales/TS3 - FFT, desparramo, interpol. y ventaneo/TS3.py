import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp

def eje_temporal (N, fs):
    
    Ts = 1/fs # t_final siempre va a ser 1 / df
    t_final = N * Ts # su inversa es la resolución espectral
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    
    return tt


def func_senoidal (tt, amp, frec, fase, v_medio, SNR):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio
    potencia_xx = np.sum (xx**2)
    
    if SNR == None:
        return xx
    
    potencia_ruido = potencia_xx / (10**(SNR/10))
    ruido = np.sqrt (potencia_ruido) * np.random.randn (len(xx))
    
    return xx + ruido


N = 1000
fs = 1000
df = fs / N

tt = eje_temporal (N, fs)
nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral

x_1 = func_senoidal (tt = tt, amp = 1, frec = (N/4)*df, fase = 0, v_medio = 0, SNR = None) 
# observar que (N/4)*df = (N/4)*(fs/N) = fs/4, por ende no importa la cantidad de muestras, siempre la frecuencia será N/4
X_1 = fft (x_1)

x_2 = func_senoidal (tt = tt, amp = 1, frec = ((N/4)+0.25)*df, fase = 0, v_medio = 0, SNR = None)
X_2 = fft (x_2)

x_3 = func_senoidal (tt = tt, amp = 1, frec = ((N/4)+0.5)*df, fase = 0, v_medio = 0, SNR = None)
X_3 = fft (x_3)


### Señal 1 ###

plt.figure (1)

plt.subplot (3, 1, 1)
#plt.plot (nn, x_1, color='black') # eje de abscisas adimensional (cantidad de muestras)
plt.plot (nn*df, x_1, color='black')
plt.title ("Señal de frecuencia N/4 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, fs/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (3, 1, 2)
#plt.plot (ff, np.abs(X), color='black') # Eje de ordenadas adimensional (N/2)
plt.plot (ff, 10*np.log10(np.abs(X_1)), color='orange') # Eje de ordenadas adimensional en dB
plt.title ("Módulo de la señal en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.subplot (3, 1, 3)
plt.plot (ff, 10*np.log10(np.abs(X_1)**2), color='orange')
plt.title ("Módulo de la señal en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()


### Señal 2 ###

plt.figure (2)

plt.subplot (2, 1, 1)
plt.plot (nn*df, x_2, color='black')
plt.title ("Señal de frecuencia N/4 + 0.25 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, fs/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff, 10*np.log10(np.abs(X_2)), color='orange')
plt.title ("Módulo de la señal en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()


### Señal 3 ###

plt.figure (3)

plt.subplot (2, 1, 1)
plt.plot (nn*df, x_3, color='black')
plt.title ("Señal de frecuencia N/4 + 0.5 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, fs/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff, 10*np.log10(np.abs(X_3)), color='orange')
plt.title ("Módulo de la señal en frecuencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
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

plt.plot (ttPadding, 10*np.log10(np.abs(XPadding)), linestyle='', marker='x')
plt.plot (ff, 10*(np.log10(np.abs(X))))
# plt.xlim (0, 500)
plt.grid (True)
plt.show ()