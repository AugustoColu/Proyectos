import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal.windows as window
import scipy.stats as st
# import sounddevice as sd
import scipy.io as sio


file_path_wav = 'silbido.wav'

try:
    # 1. Cargar el archivo .wav
    # sr: sample rate, y: array de NumPy con las muestras (enteros)
    # ¡Notá el orden de retorno: sr, y!
    sr, y = sio.wavfile.read(file_path_wav)

    print(f"Archivo .wav cargado.")
    print(f"Frecuencia de muestreo: {sr} Hz")
    print(f"Tipo de datos original: {y.dtype}") # Verás 'int16', 'int32', etc.
    print(f"Forma original del array: {y.shape}") # Si es stereo, será (n_muestras, 2)

    # 2. (Opcional) Convertir a mono si es stereo promediando los canales
    if y.ndim == 2:
        print("El audio es stereo, convirtiendo a mono.")
        y = y.mean(axis=1)

    # 3. (Recomendado) Normalizar de entero a float para el análisis
    y_float = y / np.iinfo(y.dtype).max
    
    # 4. Ahora podés usar 'y_float' y 'sr' para tus gráficos y análisis
    tt = np.linspace(0, len(y_float) / sr, num=len(y_float))
    plt.figure(figsize=(12, 4))
    plt.plot(tt, y_float)
    plt.title("Forma de Onda del Archivo .wav")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud Normalizada")
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{file_path_wav}'.")
except Exception as e:
    print(f"Ocurrió un error al procesar el archivo: {e}")
    
X = fft (y_float)

ff = np.arange (len(X)) * sr/len(X) 
plt.figure (2)
plt.plot (ff, 10*np.log10(np.abs(X)**2))
plt.xlim (0, sr/2)

plt.show()