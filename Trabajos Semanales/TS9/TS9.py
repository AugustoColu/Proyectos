import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft
import scipy.signal as sp
import scipy.signal.windows as window
import scipy.stats as st
import scipy.io as sio
import scipy.interpolate as interpol
# from pytc2.sistemas_lineales import plot_plantilla

# ------------------------------- Señal de ECG con ruido ------------------------------- #

fs_ECG = 1000

# ECG = sio.whosmat ('./ECG_TP4.mat') # devuelve una lista de variables dentro del archivo .mat (MATLAB)
mat_struct = sio.loadmat ('./ECG_TP4.mat')
# print("Variables encontradas en el archivo .mat:", mat_struct.keys()) # esto muestra las variables dentro del archivo .mat (lo miro desde el explorador de variables)

ecg_one_lead = mat_struct ['ecg_lead'].ravel() # ECG con ruido

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

ECG_cr = ecg_one_lead[0:50000].ravel()

N_ECG_cr = len (ECG_cr)

df_ECG_cr = fs_ECG / N_ECG_cr
nn_ECG_cr = np.arange (N_ECG_cr)

promedios_ECG_cr = 16
nperseg_ECG_cr = N_ECG_cr // promedios_ECG_cr

ff_ECG_cr, per_ECG_cr = sp.welch (ECG_cr, nfft = 5*nperseg_ECG_cr, fs = fs_ECG, nperseg = nperseg_ECG_cr, window = 'flattop')

energia_acum_cr = np.cumsum (per_ECG_cr)
energia_acum_cr_norm = energia_acum_cr / energia_acum_cr[-1]
corte_ECG_cr = energia_acum_cr_norm[-1] * 0.995
indice_corte_cr = int (np.where (energia_acum_cr_norm >= corte_ECG_cr)[0][0])
frec_corte_cr = ff_ECG_cr[indice_corte_cr]

plt.figure (3)

plt.subplot (2, 1, 1)
plt.plot (nn_ECG_cr, ECG_cr, color='gray')
# plt.plot (ff_ECG_cr, 10*np.log10(np.abs(per_ECG_cr))) # representación en dB
plt.title ("Señal de ECG con ruido")
plt.ylabel ("Amplitud")
plt.xlabel ("Muestras")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff_ECG_cr, per_ECG_cr, color='orange')
plt.axvline (frec_corte_cr, linestyle='--', color='gray', label=f'Frecuencia de corte = {frec_corte_cr:.2f} Hz')
plt.title ("Periodograma")
plt.ylabel ("|X|^2")
plt.xlabel ("Frecuencia [Hz]")
plt.grid (True)
plt.legend ()
plt.xlim (0, 60)

plt.tight_layout()
plt.show()

# %%

# ------------------------------------- Estimación por mediana ------------------------------------- #

s = ecg_one_lead[:50000].ravel()
b_estimador = sig.medfilt (sig.medfilt (s, 201), 601)

plt.plot (s)
plt.plot (b_estimador)
plt.grid (True)
plt.plot (s-b_estimador)

# todo artefacto producido por la NO LINEALIDAD (como las discontinuidades) generan ensanchamiento del
# ancho de banda, y puede contaminar a la señal con energía en frecuencias que antes no tenían energía 

# %%

# ------------------------------------- Estimación por Splines Cúbicos ------------------------------------- #

s = ecg_one_lead.ravel()
qrs_x = mat_struct ['qrs_detections'].flatten() - 80
qrs_y = ecg_one_lead[qrs_x]

b_estimador = interpol.CubicSpline (x = qrs_x, y = qrs_y)
x = s - b_estimador(np.arange(len(s)))

plt.figure ()
plt.plot (b_estimador(np.arange(len(s))), label='Estimador')
# plt.plot (s, label='ECG', color='black')
plt.plot (x, label='ECG filtrada')
plt.plot (qrs_x, qrs_y, marker='x', ls='', color='orchid')
plt.grid (True)
plt.legend ()

# plt.figure ()
# plt.plot (s[900000:912000], label='ECG', color='black')
# plt.plot (x[900000:912000], label='estimación')
# # plt.plot (qrs_x, qrs_y, marker='x', ls='', color='orchid')
# plt.legend ()

# un filtro derivador (resta muestras) es un pasa-altos, pues tiene un cero en DC (origen de coordenadas, plano S)
# que hace crecer al módulo a medida que w aumenta
# un promediador (suma muestras) es una filtro pasa-bajos