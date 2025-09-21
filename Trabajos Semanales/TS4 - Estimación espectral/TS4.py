import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp


def func_senoidal (tt, frec, amp, fase = 0, v_medio = 0):

    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx


R = 200
SNR = 3 # SNR en dB
amp_0 = np.sqrt(2) # amplitud en V
N = 1000
fs = 1000
df = fs / N # Hz, resolución espectral
frec_rand = df*np.random.uniform(-2,2,R)

nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral
tt = np.arange (N) / fs # vector temporal

tt_col = tt.reshape (N, 1)
tt_mat = np.tile (tt_col, reps = (1, R))

frec_rand_fila = frec_rand.reshape (1, R)
frec_mat = np.tile (frec_rand_fila, reps = (N, 1))

s_1 = func_senoidal (tt = tt_mat, amp = amp_0, frec = frec_mat + (N/4)*df)

pot_ruido = amp_0**2 / (2*10**(SNR/10))
print (f"Potencia de SNR {pot_ruido:3.1f}")

ruido_mat = np.random.normal (loc = 0, scale = np.sqrt(pot_ruido), size = (N,R))
var_ruido = np.var (ruido_mat)
print (f"Potencia de ruido -> {var_ruido:3.3f}")

x_1 = s_1 + ruido_mat # modelo de señal

flattop = sp.windows.flattop(N).reshape(N, 1)
flattop_mat = np.tile (flattop, (1, R)) 

X_1 = (1/N)*fft(x_1, axis=0)
ff_zp = np.arange (10*N) * df
# print (np.var(x_1))

plt.plot (ff, 10*np.log10(np.abs(X_1)**2))
plt.xlim (0, fs/2)
plt.grid (True)
plt.show ()