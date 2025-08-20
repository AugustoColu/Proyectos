import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

def func_senoidal (a_max, frec, fase, cant_muestras, frec_muestreo, v_medio):
    
    Ts = 1/frec_muestreo
    t_final = cant_muestras * Ts
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    xx = a_max * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return tt, xx


### Inicializo variables ###

a_max = 0
frec = 0
fase = 0
cant_muestras = 2000
frec_muestreo = 1000000
v_medio = 0

print ("\nSe toman 2000 muestras con una frecuencia de muestreo de 1MHz para todas las señales \n")

### Señal 1 ###

tt_1, ss_1 = func_senoidal (a_max = 1, frec = 2000, fase = 0, cant_muestras = 2000, frec_muestreo = 1000000, v_medio = 0)

plt.subplot (6, 2, 1)
plt.plot (tt_1, ss_1, linestyle='-', color='black')
plt.title ("Señal sinusoidal de 2 KHz")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 1 es de", np.sum (ss_1**2)/cant_muestras, "[magnitud]/seg")

### Señal 2 ###

tt_2, ss_2 = func_senoidal (a_max = 10, frec = 2000, fase = np.pi/2, cant_muestras = 2000, frec_muestreo = 1000000, v_medio = 0)

plt.subplot (6, 2, 3)
plt.plot (tt_2, ss_2, linestyle='-', color='black')
plt.title ("Sinusoidal amplificada y desfasada")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 2 es de", np.sum (ss_2**2)/cant_muestras, "[magnitud]/seg")

### Señal 3 ###

tt_3, moduladora = func_senoidal (a_max = 10, frec = 1000, fase = 0, cant_muestras = 2000, frec_muestreo = 1000000, v_medio = 0)
ss_3 = moduladora * ss_1

plt.subplot (6, 2, 5)
plt.plot (tt_3, ss_3, linestyle='-', color='black')
plt.title ("Señal modulada en amplitud")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 3 es de", np.sum (ss_3**2)/cant_muestras, "[magnitud]/seg")

### Señal 4 ###

ss_4 = ss_3 * np.square (0.75) # pues la energía (en este caso, potencia) es directamente proporcional al cuadrado de la amplitud

plt.subplot (6, 2, 7)
plt.plot (tt_1, ss_4, linestyle='-', color='black')
plt.title ("Señal modulada y recordata al 75%")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 4 es de", np.sum (ss_4**2)/cant_muestras, "[magnitud]/seg")

### Señal 5 ###

ss_5 = sp.square (2 * np.pi * 4000 * tt_1)

plt.subplot (6, 2, 9)
plt.plot (tt_1, ss_5, linestyle='-', color='black')
plt.title ("Señal cuadrada de frecuencia 4 KHz")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 5 es de", np.sum (ss_5**2)/cant_muestras, "[magnitud]/seg")

### Señal 6 ###
"""
ss_6 =

plt.subplot (6, 2, 11)
plt.plot (tt_1, ss_6, linestyle='-', color='black')
plt.title ("Pulso rectangular de 10 ms")
plt.xlabel ("Tiempo")
plt.ylabel ("Amplitud")
plt.grid (True)

print ("La potencia de la señal 6 es de", np.sum (ss_6**2)/cant_muestras, "[magnitud]/seg")
"""
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
    """
print ("Producto interno entre las señales 1 y 6 ->", np.inner (ss_1, ss_6), "-> por lo tanto", end = " ")
if (np.isclose (np.inner (ss_1, ss_6), 0)):
    print ("SON ortogonales")
else:
    print ("NO son ortogonales")
"""


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
plt.title ("Correlación entre las señales 1 y 3")
plt.grid (True)

correlacion_15 = np.correlate (ss_1, ss_5, mode = 'full')
plt.subplot (6, 2, 10)
plt.plot (np.arange (len(correlacion_15)), correlacion_15, linestyle='-', color='green')
plt.title ("Correlación entre las señales 1 y 3")
plt.grid (True)
"""
correlacion_16 = np.correlate (ss_1, ss_6, mode = 'full')
plt.subplot (6, 2, 12)
plt.plot (np.arange (len(correlacion_16)), correlacion_16, linestyle='-', color='green')
plt.title ("Correlación entre las señales 1 y 3")
plt.grid (True)
"""
plt.show ()