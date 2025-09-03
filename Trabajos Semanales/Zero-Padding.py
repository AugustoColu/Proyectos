import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros de la señal ---
fs = 100  # Frecuencia de muestreo
N_original = 16 # Número de muestras original
t = np.arange(N_original) / fs
# Frecuencia que cae entre los bins de la DFT original
f0 = 15.5
signal = np.sin(2 * np.pi * f0 * t)

# --- Cálculo de las DFTs ---

# 1. DFT normal (sin padding)
dft_normal = np.fft.fft(signal)
freq_normal = np.fft.fftfreq(N_original, 1/fs)

# 2. DFT con Zero Padding
N_padded = 256 # Longitud total después de añadir ceros
dft_padded = np.fft.fft(signal, n=N_padded)
freq_padded = np.fft.fftfreq(N_padded, 1/fs)


# --- Visualización ---
plt.figure(figsize=(12, 6))
# Graficamos la DFT normal con marcadores para ver los puntos discretos
plt.plot(freq_normal, np.abs(dft_normal), 'o-', label=f'DFT Normal ({N_original} puntos)')
# Graficamos la DFT con padding como una línea suave
plt.plot(freq_padded, np.abs(dft_padded), '-', label=f'DFT con Zero Padding ({N_padded} puntos)')

plt.title("Efecto del Zero Padding en la DFT")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(0, fs / 2)
plt.legend()
plt.grid(True)
plt.show()