import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft
import scipy.signal as sp
import scipy.signal.windows as window
import scipy.stats as st
import scipy.io as sio



frecuencias = [0, 0.1, 0.8, 35, 40, fs//2] # firwin2 me pide que empiece en 0 y termine en fs/2
ganancia_deseada = [0, 0, 1, 1, 0, 0]
# con esto doy los puntos que va a tratar de interpolar el filtro
cant_coef = 100





fir_hamming = sig.firwin2 (numtaps = cant_coef, freq = frecuencias, gain = ganancia_deseada), fs = fs

# 'numtaps' son la cantidad de coeficientes de la func. transferencia T(w)

