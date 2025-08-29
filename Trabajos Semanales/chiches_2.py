import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp

def eje_temporal (N, fs):
    
    Ts = 1/fs
    t_final = N * Ts # su inversa es la resolución espectral
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    return tt

def func_senoidal (tt, amp, frec, fase, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx


N = 1000
fs = 1000
res_espec = fs / N

tt = eje_temporal (N, fs)
x = func_senoidal (tt = )