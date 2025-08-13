import matplotlib.pyplot as plt
import numpy as np


def func_senoidal (a_max, v_medio, frec, fase, cant_muestras, frec_muestreo):
    
    tt = np.arange (0, cant_muestras, frec_muestreo) # defino una sucesión de valores para el tiempo
    xx = a_max * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return (tt, xx)


a_max = float (input ("Introducir amplitud máxima: "))
v_medio = float (input ("Introducir valor medio: "))
fase = float (input ("Introducir fase: "))
frec = float (input ("Introducir frecuencia: "))
cant_muestras = int (input ("Introducir cantidad de muestras: "))
frec_muestreo = float (input ("Introducir frecuencia de muestreo: "))

tiempo, onda = func_senoidal (a_max, v_medio, frec, fase, cant_muestras, frec_muestreo)

plt.plot (tiempo, onda, color='black')
plt.title ("Onda Senoidal")
plt.grid (True)
plt.show ()