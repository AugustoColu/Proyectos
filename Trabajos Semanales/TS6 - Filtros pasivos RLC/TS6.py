import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft
import scipy.signal as sig

# ---------------------------------- Funciones transferencia ---------------------------------- #

b_1 = [1, 0, 9]          # coeficientes del numerador
a_1 = [1, np.sqrt(2), 1] # coeficientes del denominador

b_2 = [1, 0, 1/9]
a_2 = [1, 1/5, 1]

b_3 = [1, 1/5, 1]
a_3 = [1, np.sqrt(2), 1]

# --------------------------------------- Polos y ceros --------------------------------------- #

z_1, p_1, k_1 = sig.tf2zpk (b = b_1, a = a_1)
z_2, p_2, k_2 = sig.tf2zpk (b = b_2, a = a_2)
z_3, p_3, k_3 = sig.tf2zpk (b = b_3, a = a_3)

# -------------------------------- Respuestas de módulo y fase -------------------------------- #

w_1, h_1 = sig.freqs (b = b_1, a = a_1)
w_2, h_2 = sig.freqs (b = b_2, a = a_2)
w_3, h_3 = sig.freqs (b = b_3, a = a_3)

h_1_abs = np.abs (h_1)
h_2_abs = np.abs (h_2)
h_3_abs = np.abs (h_3)

fase_1 = np.unwrap (np.angle(h_1))
fase_2 = np.unwrap (np.angle(h_2))
fase_3 = np.unwrap (np.angle(h_3))

# ------------------------------------------ Ploteos ------------------------------------------ #

# %%

plt.figure (1)

plt.plot (np.real(p_1), np.imag(p_1), ls='', marker='x', markersize=10, label='Polos')
# axes_hdl = plt.gca()

if len(z_1) > 0:
    plt.plot (np.real(z_1), np.imag(z_1), ls='', marker='o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline (0, color='k', lw=0.5)
plt.axvline (0, color='k', lw=0.5)
# unit_circle = plt.patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch (unit_circle)

plt.title ('Diagrama de Polos y Ceros de T_1 (plano S)')
plt.xlabel (r'$\Re(z)$')
plt.ylabel (r'$\Im(z)$')
plt.legend ()
plt.grid (True)

plt.figure (2)

plt.subplot (2, 1, 1)
plt.semilogx (w_1, 20*np.log10(h_1_abs))
plt.title ('Respuesta de módulo de T_1')
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Módulo [dB]')
plt.grid (True)

plt.subplot (2, 1, 2)
plt.semilogx (w_1, np.degrees(fase_1))
plt.title ('Respuesta de fase de T_1')
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Fase')
plt.grid (True)

plt.tight_layout ()
plt.show ()

# %%

plt.figure (3)

plt.plot (np.real(p_2), np.imag(p_2), ls='', marker='x', markersize=10, label='Polos')
# axes_hdl = plt.gca()

if len(z_1) > 0:
    plt.plot (np.real(z_2), np.imag(z_2), ls='', marker='o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline (0, color='k', lw=0.5)
plt.axvline (0, color='k', lw=0.5)
# unit_circle = plt.patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch (unit_circle)

plt.title ('Diagrama de Polos y Ceros de T_2 (plano S)')
plt.xlabel (r'$\Re(z)$')
plt.ylabel (r'$\Im(z)$')
plt.legend ()
plt.grid (True)

plt.figure (4)

plt.subplot (2, 1, 1)
plt.semilogx (w_2, 20*np.log10(h_2_abs))
plt.title ('Respuesta de módulo de T_2')
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Módulo [dB]')
plt.grid (True)

plt.subplot (2, 1, 2)
plt.semilogx (w_2, np.degrees(fase_2))
plt.title ('Respuesta de fase de T_2')
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Fase')
plt.grid (True)

plt.tight_layout ()
plt.show ()

# %%

plt.figure (5)

plt.plot (np.real(p_3), np.imag(p_3), ls='', marker='x', markersize=10, label='Polos')
# axes_hdl = plt.gca()

if len(z_1) > 0:
    plt.plot (np.real(z_3), np.imag(z_3), ls='', marker='o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline (0, color='k', lw=0.5)
plt.axvline (0, color='k', lw=0.5)
# unit_circle = plt.patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch (unit_circle)

plt.title ('Diagrama de Polos y Ceros de T_3 (plano S)')
plt.xlabel (r'$\Re(z)$')
plt.ylabel (r'$\Im(z)$')
plt.legend ()
plt.grid (True)

plt.figure (6)

plt.subplot (2, 1, 1)
plt.semilogx (w_3, 20*np.log10(h_3_abs))
plt.title ('Respuesta de módulo de T_3')
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Módulo [dB]')
plt.grid (True)

plt.subplot (2, 1, 2)
plt.semilogx (w_3, np.degrees(fase_3))
plt.title ('Respuesta de fase de T_3')
plt.xlabel ('Pulsación angular (rad/seg)')
plt.ylabel ('Fase')
plt.grid (True)

plt.tight_layout ()
plt.show ()