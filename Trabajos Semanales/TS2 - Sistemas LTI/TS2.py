import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp


def func_senoidal (tt, frec, amp, fase, v_medio):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx


def LTI_1 (x):
    
    N = len(x)
    y = np.zeros (N)
    y[0] = 0.03*x[0]
    y[1] = 0.03*x[1] + 0.05*x[0] + 1.5*y[0]
    for n in np.arange (2, N, 1):
        y[n] = 0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2] + 1.5*y[n-1] - 0.5*y[n-2]
    return y


def LTI_2 (x):
    
    N = len(x)
    y = np.zeros (N)
    for k in np.arange (0, 9, 1):
        y[k] = x[k]
    for n in np.arange (10, N, 1):
        y[n] = x[n] + 3*x[n-10]
    return y


def LTI_3 (x):
    
    N = len(x)
    y = np.zeros (N)
    for k in np.arange (0, 9, 1):
        y[k] = x[k]
    for n in np.arange (10, N, 1):
        y[n] = x[n] + 3*y[n-10]
    return y
        

### Inicializo variables ###

amplitud = 0
frec = 0
fase = 0
v_medio = 0
N = 500
fs = 100000
df = fs/N

tt = np.arange (0, N/fs, 1/fs) # vector de tiempos
nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector de frecuencias

print ("\nSe toman 500 muestras con una frecuencia de muestreo de 100KHz para todas las señales \n")


### Señal 1 ###

ss_1 = func_senoidal (tt = tt, frec = 2000, amp = 1, fase = 0, v_medio = 0)
y_1 = LTI_1 (x = ss_1)

plt.figure (1)

plt.subplot (2, 1, 1)
plt.plot (ff, ss_1, color='black')
plt.title ("Señal sinusoidal de 2 KHz")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (tt, y_1, color='green')
plt.title ("Salida correspondiente al sistema LTI_1")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.tight_layout ()
print ("Señal 1: potencia de entrada ->", np.sum (ss_1**2)/N, "[magnitud]/seg")
print ("         potencia de salida  ->", np.sum (y_1**2)/N, "[magnitud]/seg")
print ()


### Señal 2 ###

ss_2 = func_senoidal (tt = tt, frec = 2000, amp = 10, fase = np.pi/2, v_medio = 0)
y_2 = LTI_1 (x = ss_2)

plt.figure (2)

plt.subplot (2, 1, 1)
plt.plot (tt, ss_2, color='black')
plt.title ("Sinusoidal amplificada y desfasada")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (tt, y_2, color='green')
plt.title ("Salida correspondiente al sistema LTI_1")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.tight_layout ()
print ("Señal 2: potencia de entrada ->", np.sum (ss_2**2)/N, "[magnitud]/seg")
print ("         potencia de salida  ->", np.sum (y_2**2)/N, "[magnitud]/seg")
print ()


### Señal 3 ###

moduladora = func_senoidal (tt = tt, frec = 1000, amp = 1, fase = np.pi/2, v_medio = 0)
ss_3 = moduladora * ss_2
y_3 = LTI_1 (x = ss_3)

plt.figure (3)

plt.subplot (2, 1, 1)
plt.plot (tt, ss_3, color='black')
plt.title ("Señal modulada en amplitud")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (tt, y_3, color='black', label='salida mediante LTI')
plt.title ("Salida correspondiente al sistema LTI_1")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.tight_layout ()
print ("Señal 3: potencia de entrada ->", np.sum (ss_3**2)/N, "[magnitud]/seg")
print ("         potencia de salida  ->", np.sum (y_3**2)/N, "[magnitud]/seg")
print ()


### Señal 4 ###

recorte = 0.75
ss_4 = np.clip (ss_1, -recorte, recorte)
y_4 = LTI_1 (x = ss_4)

plt.figure (4)

plt.subplot (2, 1, 1)
plt.plot (tt, ss_4, color='black')
plt.title ("Señal modulada en amplitud")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (tt, y_4, color='green')
plt.title ("Salida correspondiente al sistema LTI_1")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.tight_layout ()
print ("Señal 4: potencia de entrada ->", np.sum (ss_4**2)/N, "[magnitud]/seg")
print ("         potencia de salida  ->", np.sum (y_4**2)/N, "[magnitud]/seg")
print ()


### Señal 5 ###

ss_5 = sp.square (2 * np.pi * 4000 * tt)
y_5 = LTI_1 (x = ss_5)

plt.figure (5)

plt.subplot (2, 1, 1)
plt.plot (tt, ss_5, color='black')
plt.title ("Señal cuadrada de frecuencia 4 KHz")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (tt, y_5, color='green')
plt.title ("Salida correspondiente al sistema LTI_1")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.tight_layout ()
print ("Señal 5: potencia de entrada ->", np.sum (ss_5**2)/N, "[magnitud]/seg")
print ("         potencia de salida  ->", np.sum (y_5**2)/N, "[magnitud]/seg")
print ()


### Señal 6 ###

pulso = np.zeros (len(tt))
duracion = 0.01
flanco_subida = 0
flanco_bajada = flanco_subida + duracion

pulso [(tt >= flanco_subida) & (tt <= flanco_bajada)] = 1
# esto hace que el vector tome el valor 1 para los índices que cumplen la condición que figura entre []
y_6 = LTI_1 (x = pulso)

plt.figure (6)

plt.subplot (2, 1, 1)
plt.plot (tt, pulso, color='black')
plt.title ("Pulso rectangular de 10 ms")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (2, 1, 2)
plt.plot (tt, y_6, color='green')
plt.title ("Salida correspondiente al sistema LTI_1")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.tight_layout ()
print ("Señal 6: potencia de entrada ->", np.sum (pulso**2)/N, "[magnitud]/seg")
print ("         potencia de salida  ->", np.sum (y_6**2)/N, "[magnitud]/seg")
print ()


### Respuesta al impulso de LTI_1 ###

I = np.zeros (len(np.arange(N)))
I[0] = 1 # de esta manera genero una delta en n=0

h_1 = LTI_1 (x = I)
# con la respuesta al impulso, puedo hallar la salida del sistema para una señal x simplemente como y=x*h (*: prod. de convolución)

yy_3 = np.convolve (ss_3, h_1, mode='full')
yy_3 = yy_3 [:len(ss_3)]

plt.figure (3)

plt.plot (tt, yy_3, linestyle='', marker='x', color='green', label='Salida mediante impulso')
plt.legend ()


### Respuestas al impulso de LTI_2 y LTI_3 ###

h_2 = LTI_2 (x = I)
h_3 = LTI_3 (x = I)
# calculo la respuesta al impulso de ambos sistemas

y_2 = np.convolve (ss_1, h_2, mode='full')
y_3 = np.convolve (ss_1, h_3, mode='full')
# calculo la salida de cada sistema de la señal 1 convolucionando con la correspondiente respuesta al impulso

y_2 = y_2 [:len(ss_1)]
y_3 = y_3 [:len(ss_1)]
# trunco los resultados al tamaño del vector ss_1

plt.figure (7)

plt.subplot (3, 1, 1)
plt.plot (tt, ss_1, color='black')
plt.title ("Señal sinusoidal de 2 KHz")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.subplot (3, 1, 2)
plt.plot (tt, y_2, color='green')
plt.title ("Salida correspondiente al sistema y[n] = x[n] + 3 . x[n−10]")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)


plt.subplot (3, 1, 3)
plt.plot (tt, y_3, color='green')
plt.title ("Salida correspondiente al sistema y[n] = x[n] + 3 . y[n−10]")
plt.xlabel ("Tiempo [seg]")
plt.ylabel ("Amplitud [V]")
plt.grid (True)

plt.tight_layout ()