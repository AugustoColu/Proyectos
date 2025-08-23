import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

def eje_temporal (N, fs):
    
    # Resolución espectral = fs / N
    # t_final siempre va a ser 1/Res. espec.
    Ts = 1/fs
    t_final = N * Ts # su inversa es la resolución espectral
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    return tt

def func_senoidal (amp, frec, fase, tt, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx


### Inicializo variables ###

amplitud = 0
frec = 0
fase = 0
N = 2000
fs = 400000
v_medio = 0

print ("\nSe toman 500 muestras con una frecuencia de muestreo de 100KHz para todas las señales \n")

tt = eje_temporal (N, fs)


### Señal 1 ###

ss_1 = func_senoidal (1, 2000, 0, tt, 0)

plt.subplot (6, 2, 1)
plt.plot (tt, ss_1, linestyle='-', color='black')
plt.title ("Señal sinusoidal de 2 KHz")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 1 es de", np.sum (ss_1**2)/N, "[magnitud]/seg")

### Señal 2 ###

ss_2 = func_senoidal (10, 2000, np.pi/2, tt, 0)

plt.subplot (6, 2, 3)
plt.plot (tt, ss_2, linestyle='-', color='black')
plt.title ("Sinusoidal amplificada y desfasada")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 2 es de", np.sum (ss_2**2)/N, "[magnitud]/seg")

### Señal 3 ###

moduladora = func_senoidal (1, 1000, np.pi/2, tt, 0)
ss_3 = moduladora * ss_2

plt.subplot (6, 2, 5)
plt.plot (tt, ss_3, linestyle='-', color='black')
plt.title ("Señal modulada en amplitud")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 3 es de", np.sum (ss_3**2)/N, "[magnitud]/seg")

### Señal 4 ###

recorte = 0.75
ss_4 = np.clip (ss_1, -recorte, recorte)

plt.subplot (6, 2, 7)
plt.plot (tt, ss_4, linestyle='-', color='black')
plt.title ("Señal modulada en amplitud")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 4 es de", np.sum (ss_4**2)/N, "[magnitud]/seg")

### Señal 5 ###

ss_5 = sp.square (2 * np.pi * 4000 * tt)

plt.subplot (6, 2, 9)
plt.plot (tt, ss_5, linestyle='-', color='black')
plt.title ("Señal cuadrada de frecuencia 4 KHz")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 5 es de", np.sum (ss_5**2)/N, "[magnitud]/seg")

### Señal 6 ###

pulso = np.zeros (len(tt))
duracion = 0.01
flanco_subida = 0
flanco_bajada = flanco_subida + duracion

pulso [(tt >= flanco_subida) & (tt <= flanco_bajada)] = 1
# esto hace que el vector tome el valor 1 para los índices que cumplen la condición que figura entre []

plt.subplot (6, 2, 11)
plt.plot (tt, pulso, linestyle='-', color='black')
plt.title ("Pulso rectangular de 10 ms")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.xlim (-0.1, 0.1)
plt.ylim (-1, 2)
plt.grid (True)

print ("La potencia de la señal 6 es de", np.sum (pulso**2)/N, "[magnitud]/seg")
print ()


### Verifico ortogonalidad entre la primera señal y las demás ###

print ("Producto interno entre las señales 1 y 2 ->", np.inner (ss_1, ss_2), "-> por lo tanto", end = " ")
if (np.isclose (np.inner (ss_1, ss_2), 0)):
    print ("SON ortogonales")
else:
    print ("NO son ortogonales")
    
print ("Producto interno entre las señales 1 y 3 ->", np.inner (ss_1, ss_3), "-> por lo tanto", end = " ")
if (np.isclose (np.inner (ss_1, ss_3), 0)):
    print ("SON ortogonales")
else:
    print ("NO son ortogonales")
    
print ("Producto interno entre las señales 1 y 4 ->", np.inner (ss_1, ss_4), "-> por lo tanto", end = " ")
if (np.isclose (np.inner (ss_1, ss_4), 0)):
    print ("SON ortogonales")
else:
    print ("NO son ortogonales")
    
print ("Producto interno entre las señales 1 y 5 ->", np.inner (ss_1, ss_5), "-> por lo tanto", end = " ")
if (np.isclose (np.inner (ss_1, ss_5), 0)):
    print ("SON ortogonales")
else:
    print ("NO son ortogonales")
    
print ("Producto interno entre las señales 1 y 6 ->", np.inner (ss_1, pulso), "-> por lo tanto", end = " ")
if (np.isclose (np.inner (ss_1, pulso), 0)):
    print ("SON ortogonales")
else:
    print ("NO son ortogonales")
    
    
### Autocorrelación de la primera señal ###

correlacion_11 = np.correlate (ss_1, ss_1, mode = 'full') # esto me devuelve un vector de la misma dimensión que ss_1
plt.subplot (6, 2, 2)
plt.plot (np.arange (len(correlacion_11)), correlacion_11, linestyle='-', color='green')
plt.title ("Autocorrelación de la señal 1")
plt.grid (True)

### Correlación entre la primera señal y las demás ###

correlacion_12 = np.correlate (ss_1, ss_2, mode = 'full')
plt.subplot (6, 2, 4)
plt.plot (np.arange (len(correlacion_12)), correlacion_12, linestyle='-', color='green')
plt.title ("Correlación entre las señales 1 y 2")
plt.grid (True)

correlacion_13 = np.correlate (ss_1, ss_3, mode = 'full')
plt.subplot (6, 2, 6)
plt.plot (np.arange (len(correlacion_13)), correlacion_13, linestyle='-', color='green')
plt.title ("Correlación entre las señales 1 y 3")
plt.grid (True)

correlacion_14 = np.correlate (ss_1, ss_4, mode = 'full')
plt.subplot (6, 2, 8)
plt.plot (np.arange (len(correlacion_14)), correlacion_14, linestyle='-', color='green')
plt.title ("Correlación entre las señales 1 y 4")
plt.grid (True)

correlacion_15 = np.correlate (ss_1, ss_5, mode = 'full')
plt.subplot (6, 2, 10)
plt.plot (np.arange (len(correlacion_15)), correlacion_15, linestyle='-', color='green')
plt.title ("Correlación entre las señales 1 y 5")
plt.grid (True)

correlacion_16 = np.correlate (ss_1, pulso, mode = 'full')
plt.subplot (6, 2, 12)
plt.plot (np.arange (len(correlacion_16)), correlacion_16, linestyle='-', color='green')
plt.title ("Correlación entre las señales 1 y 6")
plt.grid (True)

plt.figure ()


### Verificación de la igualdad 2.sen(a).sen(b) = cos(a-b) - cos(a+b) ###

w = 5000
xx_1 = np.cos (w*tt) - np.cos (3*w*tt)
xx_2 = 2 * np.sin (2*w*tt) * np.sin (w*tt)

plt.subplot (2, 1, 1)
plt.plot (tt, xx_1, linestyle='-', color='black')
plt.title ("cos (w*t) - cos (3*w*t)")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (tt, xx_2, linestyle='-', color='red')
plt.title ("2 * sin (2*w*t) * sin (w*t)")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

plt.show ()