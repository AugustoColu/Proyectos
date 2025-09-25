import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal.windows as window
import scipy.stats as st
# import sounddevice as sd
import scipy.io as sio

fs_ECG = 1000

# ECG = sio.whosmat ('./ECG_TP4.mat') # devuelve una lista de variables dentro del archivo .mat (MATLAB)
mat_struct = sio.loadmat ('./ECG_TP4.mat')
# print("Variables encontradas en el archivo .mat:", mat_struct.keys()) # esto muestra las variables dentro del archivo .mat (lo miro desde el explorador de variables)

ecg_one_lead = mat_struct ['ecg_lead']

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']


plt.figure (1)

plt.subplot (3, 1, 1)
plt.plot(ecg_one_lead[5000:12000])
plt.subplot (3, 1, 2)
plt.plot(hb_1)
plt.subplot (3, 1, 3)
plt.plot(hb_2)

plt.tight_layout ()
plt.show ()


ecg_one_lead = np.load ('ecg_sin_ruido.npy') # toma el array que se encuentra en el archivo

N_ECG = len (ecg_one_lead)

df_ECG = fs_ECG / N_ECG
nn_ECG = np.arange (N_ECG)
ff_ECG = np.arange (N_ECG) * df_ECG

cant_promedios = 10 # parámetro inversamente proporcional a la varianza 
                    # en cant_promedios=1 tengo algo muy similar a la FFT pelada
                    # se debe ajustar la cantidad de promedios según: 1) mucha varianza, me quedé corto. 2) se corre el centro de masa del espectro, me fui al pasto
nperseg = N_ECG // cant_promedios
ff_ECG, psd = sp.welch (ecg_one_lead, nfft = 10*nperseg, fs = fs_ECG, nperseg = nperseg, window='flattop') # N/nperseg es la cantidad de promedios que quiero hacer
# normalmente quiero tener un padding de al menos 1000 muestras (tomar con pinzas), en este caso depende de nperseg, ajusto en el parámetro nfft

# lo que después voy a querer hacer es integrar, para calcular el area (potencia), eso se debe hacer en veces (no en dB)

plt.figure (2)

plt.subplot (2, 1, 1)
plt.plot (nn_ECG, ecg_one_lead)
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff_ECG, psd)
# plt.plot (ff_ECG, 20*np.log10(np.abs(psd)))
plt.grid (True)
plt.xlim (0, 50)

plt.tight_layout ()
plt.show ()