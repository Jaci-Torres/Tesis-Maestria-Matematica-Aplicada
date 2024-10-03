! pip install deepxde

import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

"""# Geometría y condiciones de frontera de la EDP"""

def ecPoisson(X, Y):
  dy_xx = dde.grad.hessian(Y, X)
  return -dy_xx - np.pi ** 2 * tf.cos(np.pi * X)

def frontera(X, en_frontera):
    return en_frontera

def funcion(X):
  return np.cos(np.pi * X)

geom = dde.geometry.Interval(0, 1)

bc = dde.icbc.DirichletBC(geom, funcion, frontera)
datos = dde.data.PDE(
    geom,
    ecPoisson,
    [bc],
    num_domain=20,
    num_boundary=8,
    solution=funcion,
    num_test=100
)

"""# Modelo de la red neuronal en base a la EDP"""

red = dde.nn.FNN([1] + [30] * 2 + [1], "tanh", "Glorot uniform")

modelo = dde.Model(datos, red)
modelo.compile("adam", lr=1e-3, metrics=["l2 relative error"])

"""# Entrenamiento y resultados del modelo"""

losshistory, train_state = modelo.train(iterations=5000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X = geom.uniform_points(200, True)
Y = modelo.predict(X, operator=ecPoisson)

plt.figure(figsize=(10, 5))

plt.plot(X, Y, color='green', linestyle='-', marker='o', markersize=3, label='Residuo de Poisson')

plt.axhline(0, color='blue', linestyle='--', linewidth=1, label='Línea de referencia')

plt.xlabel("x", fontsize=8)
plt.title("Residuo de la Ecuación de Poisson", fontsize=10)
plt.legend()
plt.show()

"""# Visualización de la solución predicha vs. la solución exacta"""

Y_pred = modelo.predict(X)
Y_exact = funcion(X)

plt.figure()
plt.plot(X, Y_pred, label="Solución predicha", color='b')
plt.plot(X, Y_exact, label="Solución exacta", color='r', linestyle='--')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Solución aproximada vs. solución analítica")
plt.legend()
plt.grid(True)
plt.show()