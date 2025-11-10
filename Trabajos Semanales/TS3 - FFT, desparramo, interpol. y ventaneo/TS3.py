import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp


def eje_temporal (N, fs):
    
    Ts = 1/fs # t_final siempre va a ser 1 / df
    t_final = N * Ts # su inversa es la resolución espectral
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    
    return tt


def func_senoidal (tt, amp, frec, SNR, fase = 0, v_medio = 0):
    
    ss = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio
    pot_ss = np.mean (np.abs(ss)**2)
    
    if SNR == None:
        return ss
    
    pot_ruido = pot_ss / (10**(SNR/10))
    ruido = np.sqrt(pot_ruido) * np.random.randn(len(ss))
    
    return ss + ruido


amp_0 = 1
N = 1000
fs = 1000
df = fs/N
frec_1 = (N/4)*df
frec_2 = ((N/4)+0.25)*df
frec_3 = ((N/4)+0.5)*df

tt = eje_temporal (N, fs)
nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral

x_1 = func_senoidal (tt = tt, amp = amp_0, frec = frec_1, SNR = None)
# observar que (N/4)*df = (N/4)*(fs/N) = fs/4, por ende no importa la cantidad de muestras, siempre la frecuencia será fs/4
RMS_1 = np.sqrt (np.mean(x_1**2)) # el RMS (Root Mean Square) es el valor efectivo de la señal, se calcula como la raíz de la potencia
                                  # la potencia de una señal discreta se calcula como (1/N)*np.sum(np.abs(x_1)**2) o bien np.mean(np.abs(x_1)**2)
                                  # para señales con media = 0 (centradas), vale que STD = RMS y Varianza = Potencia
x_1 = x_1 / RMS_1 # señal de potencia normalizada, se utiliza RMS o STD porque la potencia tiene unidades de amplitud^2
X_1 = fft (x_1)
print ("Potencia de la señal 1 en tiempo     (x_1) ->", np.mean(np.abs(x_1)**2))
print ("Potencia de la señal 1 en frecuencia (X_1) ->", (1/N)*np.mean(np.abs(X_1)**2))

x_2 = func_senoidal (tt = tt, amp = amp_0, frec = frec_2, SNR = None)
x_2 = x_2 / np.std (x_2) # normalizo por el STD, idem. en este caso a normalizar por el RMS
X_2 = fft (x_2)
print ("Potencia de la señal 2 en tiempo     (x_2) ->", np.mean(np.abs(x_2)**2))
print ("Potencia de la señal 1 en frecuencia (X_2) ->", (1/N)*np.mean(np.abs(X_2)**2))

x_3 = func_senoidal (tt = tt, amp = amp_0, frec = frec_3, SNR = None)
x_3 = x_3 / np.std (x_3)
X_3 = fft (x_3)
print ("Potencia de la señal 3 en tiempo     (x_3) ->", np.mean(np.abs(x_3)**2))
print ("Potencia de la señal 1 en frecuencia (X_3) ->", (1/N)*np.mean(np.abs(X_3)**2))


### Ploteos ###

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
#plt.plot (ff, np.abs(X), color='black') # Eje de ordenadas adimensional (N/2)
plt.plot (ff, 10*np.log10(np.abs(X_1/N)**2), color='orange') # Eje de ordenadas adimensional en dB
plt.title ("Densidad Espectral de Potencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()


plt.figure (2)

plt.subplot (2, 1, 1)
plt.plot (nn, x_2, color='black')
plt.title ("Señal de frecuencia N/4 + 0.25 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, N/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff, 10*np.log10(np.abs(X_2/N)**2), color='orange')
plt.title ("Densidad Espectral de Potencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()


plt.figure (3)

plt.subplot (2, 1, 1)
plt.plot (nn, x_3, color='black')
plt.title ("Señal de frecuencia N/4 + 0.5 en tiempo")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.xlim (0, N/2) # ...visualizo hasta Nyquist
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff, 10*np.log10(np.abs(X_3/N)**2), color='orange')
plt.title ("Densidad Espectral de Potencia")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)

plt.tight_layout ()

plt.show ()


# %% Zero-Padding

zeros = np.zeros (N*9)
ff_zp = np.arange (10*N) * (fs / (10*N))

x_1zp = np.concatenate ((x_1, zeros))
X_1zp = fft (x_1zp)

x_2zp = np.concatenate ((x_2, zeros))
X_2zp = fft (x_2zp)

x_3zp = np.concatenate ((x_3, zeros))
X_3zp = fft (x_3zp)


print ("\n-------- Teorema de Parseval al implementar Zero-Padding --------")

print ("Potencia de la señal 1 en tiempo     (x_1zp) ->", np.mean(np.abs(x_1zp)**2)*10)
print ("Potencia de la señal 1 en frecuencia (X_1zp) ->", (1/N)*np.mean(np.abs(X_1zp)**2))

print ("Potencia de la señal 2 en tiempo     (x_2zp) ->", np.mean(np.abs(x_2zp)**2)*10)
print ("Potencia de la señal 1 en frecuencia (X_2zp) ->", (1/N)*np.mean(np.abs(X_2zp)**2))

print ("Potencia de la señal 3 en tiempo     (x_3zp) ->", np.mean(np.abs(x_3zp)**2)*10)
print ("Potencia de la señal 1 en frecuencia (X_3zp) ->", (1/N)*np.mean(np.abs(X_3zp)**2))


plt.figure (4)

plt.subplot (3, 1, 1)

plt.plot (ff, 10*np.log10(np.abs(X_1/N)**2), color='black', label='Resolución de 1 Hz')
plt.plot (ff_zp, 10*np.log10(np.abs(X_1zp/N)**2), color='orange', label='Zero-Padding')
plt.title ("Densidad Espectral de Potencia de la señal 1")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)
plt.legend ()

plt.subplot (3, 1, 2)

plt.plot (ff, 10*np.log10(np.abs(X_2/N)**2), color='black', label='Resolución de 1 Hz')
plt.plot (ff_zp, 10*np.log10(np.abs(X_2zp/N)**2), color='orange', label='Zero-Padding')
plt.title ("Densidad Espectral de Potencia de la señal 2")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)
plt.legend ()

plt.subplot (3, 1, 3)

plt.plot (ff, 10*np.log10(np.abs(X_3/N)**2), color='black', label='Resolución de 1 Hz')
plt.plot (ff_zp, 10*np.log10(np.abs(X_3zp/N)**2), color='orange', label='Zero-Padding')
plt.title ("Densidad Espectral de Potencia de la señal 3")
plt.xlabel ("Frecuencia [Hz]")
plt.ylabel ("[dB]")
plt.xlim (0, fs/2)
plt.grid (True)
plt.legend ()

plt.tight_layout ()
plt.show ()