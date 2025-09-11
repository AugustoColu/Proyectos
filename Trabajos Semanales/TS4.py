import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp


def gen_señal (fs, N, amp, frec, fase, v_medio, SNR):
    
    t_final = N * 1/fs
    tt = np.arange (0, t_final, 1/fs)
    
    frec_rand = np.random.uniform (-2, 2)
    frec_omega = fs/4 + frec_rand * (fs/N)
    
    ruido = np.zeros (N)
    for k in np.arange (0, N, 1):
        pot_snr = amp**2 / (2*10**(SNR/10))                                 
        ruido[k] = np.random.normal (0, pot_snr)
    
    x = amp * np.sin (frec_omega * tt) + ruido
    
    return tt, x


N = 1000
fs = 1000
df = fs / N

nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral


tt, x_1 = gen_señal (fs = fs, N = N, amp = np.sqrt(2), frec = fs/4, fase = 0, v_medio = 0, SNR = 3)
X_1 = fft (np.abs(x_1))
print (np.var(x_1))

plt.plot (np.arange(N)*df, X_1)
plt.grid (True)
plt.show ()