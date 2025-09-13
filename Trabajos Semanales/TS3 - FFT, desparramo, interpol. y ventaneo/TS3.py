import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp


def eje_temporal (N, fs):
    
    t_final = N * 1/fs
    tt = np.arange (0, t_final, 1/fs)
    
    return tt


def func_senoidal (tt, amp, frec, fase, v_medio, SNR):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio
    potencia_xx = np.sum (xx**2)
    
    if SNR == None:
        return xx
    
    N = len(tt)
    ruido = np.zeros (N)
    for k in np.arange (0, N, 1):
        pot_snr = amp**2 / (2*10**(SNR/10))                                 
        ruido[k] = np.random.normal (0, pot_snr)
    
    return xx + ruido


N = 1000
fs = 1000
df = fs / N

tt = eje_temporal (N, fs)
nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral

x_1 = func_senoidal (tt = tt, amp = 1, frec = (N/4)*df, fase = 0, v_medio = 0, SNR = None)
x_1 = x_1 / np.std(x_1) # potencia normalizada, idem. a dividir por np.sum(x_1**2)
X_1 = fft (x_1)
pot_X1 = np.mean (np.abs(X_1)**2)/N
print ("Potencia de la PSD correspondiente a la señal 1 ->", pot_X1)

x_2 = func_senoidal (tt = tt, amp = 1, frec = ((N/4)+0.25)*df, fase = 0, v_medio = 0, SNR = None)
x_2 = x_2 / np.std(x_2)
X_2 = fft (x_2)
pot_X2 = np.mean (np.abs(X_2)**2)/N
print ("Potencia de la PSD correspondiente a la señal 2 ->", pot_X2)

x_3 = func_senoidal (tt = tt, amp = 1, frec = ((N/4)+0.5)*df, fase = 0, v_medio = 0, SNR = None)
x_3 = x_3 / np.std(x_3)
X_3 = fft (x_3)
pot_X3 = np.mean (np.abs(X_3)**2)/N
print ("Potencia de la PSD correspondiente a la señal 3 ->", pot_X3)


### Señal 1 ###

plt.figure (1)

plt.subplot (2, 1, 1)
#plt.plot (nn, x_1, color='black') # eje de abscisas adimensional (cantidad de muestras)
plt.plot (nn, x_1, color='black')
plt.title ("Señal de frecuencia N/4 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, N/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (2, 1, 2)
#plt.plot (ff, np.abs(X), color='black') # eje de ordenadas adimensional (N/2)
plt.plot (ff, 10*np.log10(np.abs(X_1)**2), color='orange') # eje de ordenadas adimensional en dB
plt.title ("Densidad Espectral de Potencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()


### Señal 2 ###

plt.figure (2)

plt.subplot (2, 1, 1)
plt.plot (nn, x_2, color='black')
plt.title ("Señal de frecuencia N/4 + 0.25 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, N/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff, 10*np.log10(np.abs(X_2)**2), color='orange')
plt.title ("Densidad Espectral de Potencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()


### Señal 3 ###

plt.figure (3)

plt.subplot (2, 1, 1)
plt.plot (nn, x_3, color='black')
plt.title ("Señal de frecuencia N/4 + 0.5 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, N/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff, 10*np.log10(np.abs(X_3)**2), color='orange')
plt.title ("Densidad Espectral de Potencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()
plt.show ()


# %% Zero-Padding

zeros = np.zeros (9*N)
x_1zp = np.concatenate ((x_1, zeros)) # opción 1: concateno x_1 con ceros
X_1zp = fft (x_1zp) # transformo el vector concatenado

X_2zp = fft (x_2, 10*N) # opción 2: calculo la fft con cierta longitud (el exceso de x_2 lo completa con ceros)

X_3zp = fft (x_3, 10*N)

ttPadding = np.arange (len(X_1zp)) * (fs / (len(X_1zp)))


plt.figure (4)

plt.subplot (3, 1, 1)
plt.plot (ttPadding, 10*np.log10(np.abs(X_1zp)), linestyle='', marker='x', color='black', label='FFT')
plt.plot (ttPadding, 10*np.log10(np.abs(X_1zp)**2), color='orange', label='PSD')
plt.title ("Señal de frecuencia N/4 con Zero-Padding")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)
plt.legend ()

plt.subplot (3, 1, 2)
plt.plot (ttPadding, 10*np.log10(np.abs(X_2zp)), linestyle='', marker='x', color='black', label='FFT')
plt.plot (ttPadding, 10*np.log10(np.abs(X_2zp)**2), color='orange', label='PSD señal 2')
plt.xlim (0, fs/2)
plt.title ("Señal de frecuencia N/4 + 0.25 con Zero-Padding")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.grid (True)
plt.legend ()

plt.subplot (3, 1, 3)
plt.plot (ttPadding, 10*np.log10(np.abs(X_3zp)), linestyle='', marker='x', color='black', label='FFT')
plt.plot (ttPadding, 10*np.log10(np.abs(X_3zp)**2), color='orange', label='PSD señal 3')
plt.xlim (0, fs/2)
plt.title ("Señal de frecuencia N/4 + 0.5 con Zero-Padding")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.grid (True)
plt.legend ()

plt.tight_layout ()
plt.show ()