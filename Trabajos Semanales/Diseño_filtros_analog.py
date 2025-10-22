import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft

# -------------------------------------- Plantilla de diseño -------------------------------------- #

wp = 1 # frecuencia de corte/paso (rad/seg)
ws = 5 # frecuencia de stop/detenida (rad/seg)

alpha_p = 1 # atenuación máxima a la wp, alpha_max, pérdida en banda de paso 
alpha_s = 40 # atenuación mínima a la ws, alpha_min, mínima atenuación en banda de paso

f_aprox = 'butter'     # aproxima módulo
# f_aprox = 'cheby1'   # aproxima módulo
# f_aprox = 'cheby2'   # aproxima módulo
# f_aprox = 'ellip'    # aproxima módulo
# f_aprox = 'cauer'    # aproxima módulo
# f_aprox = 'bessel'   # aproxima fase

# ---------------------------------- Diseño del filtro analógico ---------------------------------- #

b, a = sig.iirdesign (wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = True, ftype = f_aprox, output = 'ba')

# ------------------------------------ Respuesta en frecuencia ------------------------------------ #

w, h = sig.freqs (b = b, a = a) # observar en el explorador de variables que tomó 200 frecuencias logarítmicamente espaciadas entre -1=log(0.1) y 1=log(10)

fase = np.unwrap(np.angle(h))
demora = -np.diff(fase) / np.diff(w)

z, p, k = sig.tf2zpk (b = b, a = a) # pasaje a zpk para visualizar polos y ceros

# -------------------------------------------- Ploteos -------------------------------------------- #

plt.figure (1)

plt.plot (np.real(p), np.imag(p), ls='', marker='x')
plt.grid (True)
plt.xlim (-1, 0.1)
plt.show ()

if len(z)>0:
    plt.plot (np.real(z), np.imag(z))



plt.figure (2)

plt.subplot (3, 1, 1)
plt.semilogx (w, 20*np.log10(np.abs(h)))
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Respuesta de módulo')
plt.grid (True)

plt.subplot (3, 1, 2)
plt.semilogx (w, np.degrees(fase))
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Respuesta de fase')
plt.grid (True)

plt.subplot (3, 1, 3)
plt.semilogx (w[:-1], demora)
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Retardo de grupo (demora)')
plt.grid (True)

plt.tight_layout ()
plt.show ()


# ---------------------------------------- Matriz de SOS's ---------------------------------------- #

sos = sig.tf2sos (b = b, a = a)