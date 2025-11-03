import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft
import scipy.signal as sig

# --------------------------------- Definición de funciones --------------------------------- #

def filtro_digital (b, a, w_num, h_num):
    
    w, h = sig.freqz (b = b, a = a)
    fase = np.unwrap(np.angle(h))
    fase_num = np.unwrap(np.angle(h_num))
    
    plt.figure ()

    plt.subplot (2, 1, 1)
    plt.plot (w, 20*np.log10(np.abs(h)))
    plt.plot (w_num, 20*np.log10(np.abs(h_num)), ls='--', color='orange')
    plt.title ('Respuesta de módulo')
    plt.xlabel ('Pulsación angular [rad/muestra]')
    plt.ylabel ('|H(w)| [dB]')
    plt.ylim (-50, 20)
    plt.grid (True)

    plt.subplot (2, 1, 2)
    plt.plot (w, np.degrees(fase))
    plt.plot (w_num, np.degrees(fase_num), ls='--', color='orange')
    plt.title ('Respuesta de fase')
    plt.xlabel ('Pulsación angular [rad/muestra]')
    plt.ylabel ('Fase [°]')
    plt.grid (True)

    plt.tight_layout ()
    plt.show ()
    
    return
    
# --------------------------------- Definición de parámetros --------------------------------- #

coef_Ba = [1, 1, 1, 1]
coef_Bb = [1, 1, 1, 1, 1]
coef_Bc = [1, -1]
coef_Bd = [1, 0, -1]

coef_A = 1 # pues los 4 son filtros de respuesta finita

w = np.linspace (0, np.pi, 1000)

# --------------------------------- Verificación numérica --------------------------------- #

h_num_a = 2 * np.exp(-1j*1.5*w) * (np.cos(1.5*w) + np.cos(0.5*w))
h_num_b = np.exp(-1j*2*w) * (1 + 2*np.cos(w) + 2*np.cos(2*w))
h_num_c = 2 * np.sin(w/2) * np.exp(1j*(np.pi/2-w/2))
h_num_d = 2 * np.sin(w) * np.exp(1j*(np.pi/2-w))

# --------------------------------- Implementación --------------------------------- #

filtro_digital (b = coef_Ba, a = coef_A, w_num = w, h_num = h_num_a)
filtro_digital (b = coef_Bb, a = coef_A, w_num = w, h_num = h_num_b)
filtro_digital (b = coef_Bc, a = coef_A, w_num = w, h_num = h_num_c)
filtro_digital (b = coef_Bd, a = coef_A, w_num = w, h_num = h_num_d)