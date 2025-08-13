import matplotlib.pyplot as plt
import numpy as np


def func_senoidal (a_max, v_medio, frec, fase, t_final, t_muestreo):
    
    tt = np.arange(0, t_final, t_muestreo) # defino una sucesión de valores para el tiempo
    xx = a_max * np.sin(2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    
    return tt, xx

a_max = 10 #float (input ("Introducir amplitud máxima: "))
v_medio = 2 #float (input ("Introducir valor medio: "))
fase = 0 #float (input ("Introducir fase: "))
frec = 10 #float (input ("Introducir frecuencia: "))
cant_muestras = 200 #int (input ("Introducir cantidad de muestras: "))
frec_muestreo = 100 #float (input ("Introducir frecuencia de muestreo: "))

t_muestreo = 1/frec_muestreo
t_final = cant_muestras * t_muestreo

tiempo, onda = func_senoidal (a_max, v_medio, frec, fase, t_final, t_muestreo)

plt.plot (tiempo, onda, linestyle='-', color='black')
plt.title ("Onda Senoidal")
plt.grid (True)
plt.show ()