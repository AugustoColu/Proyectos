import matplotlib.pyplot as plt
import numpy as np


def func_senoidal (a_max, frec, fase, cant_muestras, frec_muestreo, v_medio):
    
    Ts = 1/frec_muestreo
    t_final = cant_muestras * Ts
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    xx = a_max * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    
    return tt, xx

### Inicializo variables en cero ###

a_max = 0
frec = 0
fase = 0
cant_muestras = 0
frec_muestreo = 0
v_medio = 0

### Señal 1 ###

tt_1, ss_1 = func_senoidal (a_max = 1, frec = 2000, fase = 0, cant_muestras = 2000, frec_muestreo = 1000000, v_medio = 0)

plt.subplot (2, 1, 1)
plt.plot (tt_1, ss_1, linestyle='-', color='black')
plt.title ("Onda Senoidal")
#plt.xlabel ("Timepo")
#plt.ylabel ("Volts")
plt.grid (True)

### Señal 2 ###

tt_2, ss_2 = func_senoidal (a_max = 10, frec = 2000, fase = np.pi/2, cant_muestras = 2000, frec_muestreo = 1000000, v_medio = 0)

plt.subplot (2, 1, 2)
plt.plot (tt_2, ss_2, linestyle='-', color='black')
plt.title ("Onda Senoidal")
#plt.xlabel ("Timepo")
#plt.ylabel ("Volts")
plt.grid (True)

plt.show ()

