import matplotlib.pyplot as plt
import numpy as np

fs = 500 # defino la frecuencia de la onda
N = fs
df = fs/N
paso = 1/fs # defino el paso
tiempo_simu = N*paso

tt = np.arange (0, tiempo_simu, paso) # defino una sucesión de puntos en el tiempo

fx = 10
ax = 2

xx = ax * np.sin (2*np.pi*fx*tt) # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
# xx tendrá la misma dimensión que tt

plt.plot (tt, xx)