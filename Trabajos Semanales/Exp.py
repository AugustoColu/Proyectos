import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Plantilla de diseño ---

wp = 2  # frecuencia de corte/paso (rad/s)
ws = 30  # frecuencia de stop/detenida (rad/s)

alpha_p = 3  # atenuación máxima a la wp, alfa_max, pérdidas en banda de paso
alpha_s = 40  # atenuación mínima a la ws, alfa_min, mínima atenuación requerida
             # en banda de paso

# Aprox módulo
f_aprox= 'butter'
# f_aprox= 'cheby1'
# f_aprox= 'cheby2'
# f_aprox= 'cauer'

# Aprox fase
# f_aprox= 'bessel'

# --- Diseño del filtro analógico ---
# b, a = signal.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
#                         analog=True, ftype= f_aprox, output='ba' )

b = [1, 0, 4]
a = [1, 2*np.sqrt(2), 4]

# %%


# --- Respuesta en frecuencia ---
w, h = signal.freqs(b, a, worN=np.logspace(-1, 2, 1000))  # 10 Hz a 1 MHz aprox.
# w, h = signal.freqs(b, a)  # Calcula la respuesta en frecuencia del filtro

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(h))
# Retardo de grupo = -dφ/dω
gd = -np.diff(phase) / np.diff(w)

# --- Polos y ceros ---
z, p, k = signal.tf2zpk(b, a)

# --- Gráficas ---
# plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.semilogx(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular  [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(2,2,2)
plt.semilogx(w, np.degrees(phase), label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular  [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(2,2,3)
plt.semilogx(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular  [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos' )
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# %%

numz, denz = signal.bilinear(b, a, fs = 1)


# --- Respuesta en frecuencia ---
w, h = signal.freqz(numz, denz)  # 10 Hz a 1 MHz aprox.

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(h))
# Retardo de grupo = -dφ/dω
gd = -np.diff(phase) / np.diff(w)

# --- Polos y ceros ---
z, p, k = signal.tf2zpk(numz, denz)

# --- Gráficas ---
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w/ np.pi, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular  [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase|
plt.subplot(3,1,2)
plt.plot(w/ np.pi, np.degrees(phase), label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular  [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1]/ np.pi, gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular  [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Diagrama de polos y ceros
plt.figure(figsize=(12,10))

plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos' )
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano z)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.legend()
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()