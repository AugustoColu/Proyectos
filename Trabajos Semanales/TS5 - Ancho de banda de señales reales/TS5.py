import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal.windows as window
import scipy.stats as st
# import sounddevice as sd
import scipy.io as sio


# ----------------------- Lecutra de ECG ----------------------- #
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

# ---------------------------------------------------- ECG con ruido ---------------------------------------------------- #

ecg_recorte = ecg_one_lead[670000:700000].ravel()

N_ECG_cr = len (ecg_recorte)

df_ECG_cr = fs_ECG / N_ECG_cr
nn_ECG_cr = np.arange (N_ECG_cr)
ff_ECG_cr = np.arange (N_ECG_cr) * df_ECG_cr

cant_promedios_cr = 12
nperseg_cr = N_ECG_cr // cant_promedios_cr

ff_ECG_cr, psd_ECG_cr = sp.welch (ecg_recorte, nfft = 10*nperseg_cr, fs = fs_ECG, nperseg = nperseg_cr, window='flattop')


# ---------------------------------------------------- ECG sin ruido ---------------------------------------------------- #

ecg_sin_ruido = np.load ('ecg_sin_ruido.npy') # toma el array que se encuentra en el archivo

N_ECG = len (ecg_sin_ruido)

df_ECG = fs_ECG / N_ECG
nn_ECG = np.arange (N_ECG)

cant_promedios = 12 # parámetro inversamente proporcional a la varianza 
                    # en cant_promedios=1 tengo algo muy similar a la FFT pelada
                    # se debe ajustar la cantidad de promedios según: 1) mucha varianza, me quedé corto. 2) se corre el centro de masa del espectro, me fui al pasto
nperseg = N_ECG // cant_promedios
ff_ECG, psd_ECG = sp.welch (ecg_sin_ruido, nfft = 10*nperseg, fs = fs_ECG, nperseg = nperseg, window='flattop') # N/nperseg es la cantidad de promedios que quiero hacer
# normalmente quiero tener un padding de al menos 1000 muestras (tomar con pinzas), en este caso depende de nperseg, ajusto con el parámetro nfft

energia_acum = np.cumsum (psd_ECG) # esto devuelve un vector de sumas acumuladas, el area que estoy buscando vendría a ser el último valor
energia_acum_norm = energia_acum / energia_acum[-1] # con [-1] accedo al último valor del vector
corte = energia_acum_norm[-1] * 0.995
indice_corte = int (np.where (energia_acum_norm >= corte)[0][0]) # con [0][0] me devuelve el primer valor que cumple con la condición
frec_corte = ff_ECG[indice_corte]


plt.figure (2)

plt.subplot (2, 1, 1)
plt.plot (nn_ECG, ecg_sin_ruido)
plt.title ("Señal de ECG")
plt.ylabel ("Amplitud")
plt.xlabel ("Muestras")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (ff_ECG, psd_ECG)
plt.plot (ff_ECG_cr, psd_ECG_cr)
plt.axvline (frec_corte, linestyle='--', color='orange', label=f'Frecuencia de corte = {frec_corte} Hz')
# plt.plot (ff_ECG, 10*np.log10(np.abs(psd)))
plt.title ("ECG sin ruido por Método de Welch")
plt.ylabel ("|X|^2")
plt.xlabel ("Frecuencia [Hz]")
plt.grid (True)
plt.legend ()
plt.xlim (0, 50)

plt.tight_layout ()
plt.show ()

# %%


# ----------------------------------- Lecutra de audios ----------------------------------- #

fs_1, wav_data_1 = sio.wavfile.read ('prueba psd.wav')
fs_2, wav_data_2 = sio.wavfile.read ('silbido.wav')
fs_3, wav_data_3 = sio.wavfile.read ('la cucaracha.wav')

