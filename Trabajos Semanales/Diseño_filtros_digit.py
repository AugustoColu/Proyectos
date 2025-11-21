import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft
import scipy.signal as sp
import scipy.signal.windows as window
import scipy.stats as st
import scipy.io as sio
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

# -------------------------------------- Diseño de filtro IIR -------------------------------------- #

fs = 1000

wp = [0.8, 35] # frecuencia de corte/paso (rad/seg)
ws = [0.1, 40] # frecuencia de stop/detenida (rad/seg)

# alpha_p = 1 # atenuación máxima a la wp, alpha_max, pérdida en banda de paso 
# alpha_s = 40 # atenuación mínima a la ws, alpha_min, mínima atenuación en banda de paso

# si utilizo sig.sosfiltfilt (filtro dos veces y sincroniza, algo así), debo atenuar la mitad (termina sumando el total)
alpha_p = 1/2
alpha_s = 40/2

# f_aprox = 'butter'   # aproxima módulo
# f_aprox = 'cheby1'   # aproxima módulo
# f_aprox = 'cheby2'   # aproxima módulo
f_aprox = 'cauer'    # aproxima módulo
# f_aprox = 'bessel'   # aproxima fase

# ---------------------------------- Diseño del filtro digital ---------------------------------- #

mat_sos = sig.iirdesign (wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output = 'sos', fs = fs)
# es IMPORTANTÍSIMO trabajar con secciones de 2° orden (SOS) para acortar la recursión (estabilidad, ???)

# ------------------------------------ Respuesta en frecuencia ------------------------------------ #

w, h = sig.freqz_sos (sos = mat_sos, worN=np.logspace(-2, 1.9, 1000), fs = fs)
w_rad = w / (fs*np.pi/2) # pues el omega devuelto tiene unidades de rad/muestra y va de [0, pi), lo normalizo por ws en nyquist

fase = np.unwrap(np.angle(h))
demora = -np.diff(fase) / np.diff(w_rad)

z, p, k = sig.sos2zpk (mat_sos) # pasaje a zpk para visualizar polos y ceros

# --------------------------------------- Respuesta del filtro --------------------------------------- #

# plt.figure (1)

# plt.figure (figsize=(10,10))
# plt.plot (np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
# axes_hdl = plt.gca()

# if len(z) > 0:
#     plt.plot (np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
# plt.axhline (0, color='k', lw=0.5)
# plt.axvline (0, color='k', lw=0.5)
# unit_circle = plt.patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch (unit_circle)

# plt.axis ([-1.1, 1.1, -1.1, 1.1])
# plt.title ('Diagrama de Polos y Ceros (plano Z)')
# plt.xlabel (r'$\Re(z)$')
# plt.ylabel (r'$\Im(z)$')
# plt.legend ()
# plt.grid (True)

# los escalones en la fase se deben a los ceros

plt.figure (2)

plt.subplot (3, 1, 1)
plt.plot (w, 20*np.log10(np.abs(h)), label=f_aprox)
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de módulo')
plt.legend ()
plt.grid (True)

plt.subplot (3, 1, 2)
plt.plot (w, np.degrees(fase))
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de fase')
plt.grid (True)

plt.subplot (3, 1, 3)
plt.plot (w[:-1], demora)
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Retardo de grupo (demora) [muestras]')
plt.grid (True)

plt.tight_layout ()
plt.show ()

# %%

# ----------------------------- Implementación del filtro ----------------------------- #

# ecg_filt = sig.sosfilt (mat_sos, ECG_cr)
ecg_filt = sig.sosfiltfilt (mat_sos, ecg_one_lead)

plt.figure (4)

plt.plot (ecg_one_lead, label='ECG con ruido')
plt.plot (ecg_filt, label=f_aprox)
plt.legend ()
plt.grid (True)

# el desfase que se observa entre las señales es producto del retardo de grupo (demora)
# ejemplo: 'butter'
# -> a bajas frecuencias tiene un retardo altísimo => observar que las "figuras" de menos frecuencia (más espaciadas, lentas) de la señal
#    tienen una "deformación" y están corridas aproximadamente 12000 muestras
# -> observar que entre picos hay, aproximadamente, 150 muestras => en el gráfico de respuesta de retardo del filtro corresponde a un ancho
#    de entre 5 y 30 Hz. Por lo tanto, a ese rango de frecuencias, la señal filtrada será "más fiel" a la original

# un buen filtro debe ser inocuo (poca alteración de la señal) en banda de paso y destructivo en banda de rechazo

# los picos de la ECG que se asemejan a un pulso, dan a la salida del filtro algo muy similar a la RESPUESTA AL IMPULSO
# pues el filtro no deja de ser un sistema LTI
# esta respuesta aparece en ambas direcciones, pues estoy filtrando bidireccionalmente, debido a filtfilt

# %%

# -------------------------------------- Diseño de filtro FIR (método de ventanas) -------------------------------------- #

fs = 1000
frecuencias = [0, 0.1, 0.8, 35, 35.7, fs//2] # firwin2 me pide que empiece en 0 y termine en fs/2
ganancia_deseada = [0, 0, 1, 1, 0, 0]
# con esto doy los puntos que va a tratar de interpolar el filtro
cant_coef = 2000
retardo = (cant_coef - 1) // 2 # como es NO entero, debo 

alpha_p = 1
alpha_s = 40

fir_rect = sig.firwin2 (numtaps = cant_coef, freq = frecuencias, gain = ganancia_deseada, window = 'boxcar', fs = fs, nfreqs = int((np.ceil(np.sqrt(cant_coef*2)**2))-1))
# 'numtaps' es la cantidad de coeficientes de la func. transferencia T(w)
# la función 'firwin2' devuelve los coeficientes 'b' del filtro FIR
# observar que al graficar 'fir_hamming' se obtiene la respuesta al impuslo del filtro
# los coef. 'b' SON la respuesta al impulso, analizo si es par o impar
# con esto concluyo dónde hay ceros topológicos, y si es que me sirve para la ganancia deseada a cada frecuencia (fs/2 = pi)

w, h = sig.freqz (b = fir_rect, worN=np.logspace(-2, 2, 1000), fs = fs)

w_rad = w / (fs/2) * np.pi #### NO ENTIENDO ESTO ####

fase = np.unwrap(np.angle(h))
demora = -np.diff(fase) / np.diff(w_rad)

z, p, k = sig.sos2zpk (sig.tf2sos(b = fir_rect, a = 1)) # pasaje a zpk para visualizar polos y ceros


# plt.figure (figsize=(10,10))

# plt.plot (np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
# axes_hdl = plt.gca()

# if len(z) > 0:
#     plt.plot (np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
# plt.axhline (0, color='k', lw=0.5)
# plt.axvline (0, color='k', lw=0.5)
# unit_circle = plt.patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch (unit_circle)

# plt.axis ([-1.1, 1.1, -1.1, 1.1])
# plt.title ('Diagrama de Polos y Ceros (plano Z)')
# plt.xlabel (r'$\Re(z)$')
# plt.ylabel (r'$\Im(z)$')
# plt.legend ()
# plt.grid (True)

plt.figure ()

plt.subplot (3, 1, 1)
# plot_plantilla (filter_type = 'bandpass', fpass = (0.8, 35), ripple = alpha_p*2, fstop = (0.1, 35.7), attenuation = alpha_s*2, fs = fs)
plt.plot (w, 20*np.log10(np.abs(h)))
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de módulo')
plt.legend ()
plt.grid (True)

plt.subplot (3, 1, 2)
plt.plot (w, np.degrees(fase))
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de fase')
plt.grid (True)

plt.subplot (3, 1, 3)
plt.plot (w[:-1], demora)
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Demora [muestras]')
plt.grid (True)

plt.tight_layout ()
plt.show ()



ecg_filt_fir = sig.lfilter (b = fir_rect, a = 1, x = ecg_one_lead)

plt.figure (4)

plt.plot (ecg_one_lead, label='ECG con ruido')
plt.plot (ecg_filt_fir, label=f_aprox)
plt.legend ()
plt.grid (True)

# a diferencia del IIR, no fue necesario filtrar dos veces (filtfilt) para compensar la demora

# %%

# -------------------------------------- Diseño de filtro FIR (método de cuad. mínimos) -------------------------------------- #

fs = 1000
frecuencias = [0, 0.4, 0.8, 35, 35.4, fs/2] # acá se "predistorsionó la plantilla" para dar más holgura a las verdaderas
                                            # frecuencias de la plantilla (alternativa para no aumentar coeficientes)
ganancia_deseada = [0, 0, 1, 1, 0, 0]
peso = [2, 1, 12] # acá tuve que "pesar", darle más importancia, a la transición de baja porque no cumplía con la plantilla
# con esto doy los puntos que va a tratar de interpolar el filtro
cant_coef = 2501 # firls me pide una cantidad par, pues solo diseña filtros de tipo I
retardo = (cant_coef - 1) // 2 # como es NO entero, debo 

alpha_p = 1
alpha_s = 40

fir_ls = sig.firls (numtaps = cant_coef, bands = frecuencias, desired = ganancia_deseada, weight = peso, fs = fs)

w, h = sig.freqz (b = fir_ls, worN=np.logspace(-2, 2, 2000), fs = fs)

w_rad = w / (fs/2) * np.pi #### NO ENTIENDO ESTO ####

fase = np.unwrap(np.angle(h))
demora = -np.diff(fase) / np.diff(w_rad)

z, p, k = sig.sos2zpk (sig.tf2sos(b = fir_rect, a = 1)) # pasaje a zpk para visualizar polos y ceros

plt.figure ()

plt.subplot (3, 1, 1)
# plot_plantilla (filter_type = 'bandpass', fpass = (0.8, 35), ripple = alpha_p*2, fstop = (0.1, 35.7), attenuation = alpha_s*2, fs = fs)
plt.plot (w, 20*np.log10(np.abs(h)))
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de módulo')
plt.legend ()
plt.grid (True)

plt.subplot (3, 1, 2)
plt.plot (w, np.degrees(fase))
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de fase')
plt.grid (True)

plt.subplot (3, 1, 3)
plt.plot (w[:-1], demora)
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Demora [muestras]')
plt.grid (True)

plt.tight_layout ()
plt.show ()

# %%

# -------------------------------------- Diseño de filtro FIR (método de Peaks-McClellan) -------------------------------------- #

fs = 1000
frecuencias = [0, 0.1, 0.8, 35, 35.7, fs/2] # para este método debo hacer "simétricas" las transiciones, sino no converge
ganancia_deseada = [0, 1, 0] # Peaks-McClellan toma la ganancia de a sectores
# con esto doy los puntos que va a tratar de interpolar el filtro
cant_coef = 1500
retardo = (cant_coef - 1) // 2 # como es NO entero, debo 

alpha_p = 1
alpha_s = 40

fir_pm = sig.remez (numtaps = cant_coef, bands = frecuencias, desired = ganancia_deseada, fs = fs)

w, h = sig.freqz (b = fir_pm, worN=np.logspace(-2, 2, 2500), fs = fs)

w_rad = w / (fs/2) * np.pi #### NO ENTIENDO ESTO ####

fase = np.unwrap(np.angle(h))
demora = -np.diff(fase) / np.diff(w_rad)

z, p, k = sig.sos2zpk (sig.tf2sos(b = fir_pm, a = 1)) # pasaje a zpk para visualizar polos y ceros

plt.figure ()

plt.subplot (3, 1, 1)
# plot_plantilla (filter_type = 'bandpass', fpass = (0.8, 35), ripple = alpha_p*2, fstop = (0.1, 35.7), attenuation = alpha_s*2, fs = fs)
plt.plot (w, 20*np.log10(np.abs(h)))
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de módulo')
plt.legend ()
plt.grid (True)

plt.subplot (3, 1, 2)
plt.plot (w, np.degrees(fase))
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Respuesta de fase')
plt.grid (True)

plt.subplot (3, 1, 3)
plt.plot (w[:-1], demora)
plt.xlabel ('Frecuencia [Hz]')
plt.ylabel ('Demora [muestras]')
plt.grid (True)

plt.tight_layout ()
plt.show ()