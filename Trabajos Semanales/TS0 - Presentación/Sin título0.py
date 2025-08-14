
import matplotlib.pyplot as plt
import numpy as np


def func_senoidal (a_max, frec, fase, t_final, Ts, v_medio):
    
    tt = np.arange (0, t_final, Ts) # defino una sucesión de valores para el tiempo
    xx = a_max * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    
    return tt, xx

a_max = 10 #float (input ("Introducir amplitud máxima: "))
v_medio = 0 #float (input ("Introducir valor medio: "))
fase = 0 #float (input ("Introducir fase: "))
frec = 10 #float (input ("Introducir frecuencia: "))
cant_muestras = 1000 #int (input ("Introducir cantidad de muestras: "))
frec_muestreo = 1000 #float (input ("Introducir frecuencia de muestreo: "))
delta_f=frec_muestreo/cant_muestras
Ts = 1/frec_muestreo
t_final = cant_muestras * Ts
tiempo_total=1/delta_f

# tt_rad = 2 * np.pi * np.arange (0, t_final, Ts), eje en radianes
tiempo, onda = func_senoidal (a_max, frec, fase, t_final, Ts, v_medio)

plt.plot (tiempo, onda, linestyle='-', color='black')
plt.title ("Onda Senoidal")
#plt.xlabel ("Timepo")
#plt.ylabel ("Volts")
plt.grid (True)
plt.show ()
