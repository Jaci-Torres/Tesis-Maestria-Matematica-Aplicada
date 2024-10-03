!pip install deepxde

import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Geometría y condiciones de frontera el problema

rho = 1
mu = 1
u_in = 1.85
h = 1.95
L = 5


def frontera_pared(X, en_frontera):
    return np.logical_and(np.logical_or(np.isclose(X[1], -h / 2), np.isclose(X[1], h / 2)), en_frontera)

def frontera_entrada(X, en_frontera):
    return np.logical_and(np.isclose(X[0], -L / 2), en_frontera)

def frontera_salida(X, en_frontera):
    return np.logical_and(np.isclose(X[0], L / 2), en_frontera)

#Definición de la ecuación de Navier-Stokes en 2D
def NavierStokes(X, Y):
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)
    du_xx = dde.grad.hessian(Y, X, i=0, j=0, component=0)
    du_yy = dde.grad.hessian(Y, X, i=1, j=1, component=0)
    dv_xx = dde.grad.hessian(Y, X, i=0, j=0, component=1)
    dv_yy = dde.grad.hessian(Y, X, i=1, j=1, component=1)

    NavierStokes_u = Y[:, 0:1]*du_x + Y[:, 1:2]*du_y + 1/rho * dp_x - (mu/rho)*(du_xx + du_yy)
    NavierStokes_v = Y[:, 0:1]*dv_x + Y[:, 1:2]*dv_y + 1/rho * dp_y - (mu/rho)*(dv_xx + dv_yy)
    NavierStokes_cont = du_x + dv_y

    return [NavierStokes_u, NavierStokes_v, NavierStokes_cont]


# Geometría del problema

geom = dde.geometry.Rectangle(xmin=[-L / 2, -h / 2], xmax=[L / 2, h / 2])

# Condiciones de frontera

frontera_pared_u = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=0)
frontera_pared_v = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=1)

frontera_entrada_u = dde.DirichletBC(geom, lambda X: u_in, frontera_entrada, component=0)
frontera_entrada_v = dde.DirichletBC(geom, lambda X: 0., frontera_entrada, component=1)

frontera_salida_p = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=2)
frontera_salida_v = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=1)


# Modelo de la PINN en base a la ecuación de Navier-Stokes en 2D

datos = dde.data.PDE(geom,
    NavierStokes,
    [frontera_pared_u, frontera_pared_v, frontera_entrada_u, frontera_entrada_v, frontera_salida_p, frontera_salida_v],
    num_domain = 2600,
    num_boundary = 400,
    num_test = 10000
)

# Visualización de los puntos de entrenamiento
plt.figure(figsize=(5, 4))
plt.scatter(datos.train_x_all[:, 0], datos.train_x_all[:, 1], s=0.5)
plt.xlabel('Longitud a lo largo del eje x')
plt.ylabel('Distancia desde el eje medio del canal')
plt.show()


# Arquitectura de la red neuronal informada por la física

pinn = dde.maps.FNN([2] + 4 * [50] + [3], "tanh", "Glorot uniform")

# Creación y entrenamiento del modelo
modelo = dde.Model(datos, pinn)
modelo.compile("adam", lr=1e-3)

losshistory, train_state = modelo.train(iterations=10000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)



# Gráfica de resultados del modelo
samples = geom.random_points(500000)
resultado = modelo.predict(samples)
colores = [[0, 1.5], [-0.3, 0.3], [0, 35]]
titulos = ['Predicción para u', 'Predicción para v', 'Predicción para p']

for graf in range(3):
    plt.figure(figsize=(15, 4))
    plt.scatter(samples[:, 0], samples[:, 1], c=resultado[:, graf], cmap='jet', s=2)
    plt.colorbar()
    plt.clim(colores[graf])
    plt.title(titulos[graf])
    plt.tight_layout()
    plt.show()

# Geometría y condiciones de frontera el problema

rho = 1
mu = 1
u_in = 1.85
h = 1.95
L = 5


def frontera_pared(X, en_frontera):
    return np.logical_and(np.logical_or(np.isclose(X[1], -h / 2), np.isclose(X[1], h / 2)), en_frontera)

def frontera_entrada(X, en_frontera):
    return np.logical_and(np.isclose(X[0], -L / 2), en_frontera)

def frontera_salida(X, en_frontera):
    return np.logical_and(np.isclose(X[0], L / 2), en_frontera)

#Definición de la ecuación de Navier-Stokes en 2D
def NavierStokes(X, Y):
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)
    du_xx = dde.grad.hessian(Y, X, i=0, j=0, component=0)
    du_yy = dde.grad.hessian(Y, X, i=1, j=1, component=0)
    dv_xx = dde.grad.hessian(Y, X, i=0, j=0, component=1)
    dv_yy = dde.grad.hessian(Y, X, i=1, j=1, component=1)

    NavierStokes_u = Y[:, 0:1]*du_x + Y[:, 1:2]*du_y + 1/rho * dp_x - (mu/rho)*(du_xx + du_yy)
    NavierStokes_v = Y[:, 0:1]*dv_x + Y[:, 1:2]*dv_y + 1/rho * dp_y - (mu/rho)*(dv_xx + dv_yy)
    NavierStokes_cont = du_x + dv_y

    return [NavierStokes_u, NavierStokes_v, NavierStokes_cont]


# Geometría del problema

geom = dde.geometry.Rectangle(xmin=[-L / 2, -h / 2], xmax=[L / 2, h / 2])

# Condiciones de frontera

frontera_pared_u = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=0)
frontera_pared_v = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=1)

frontera_entrada_u = dde.DirichletBC(geom, lambda X: u_in, frontera_entrada, component=0)
frontera_entrada_v = dde.DirichletBC(geom, lambda X: 0., frontera_entrada, component=1)

frontera_salida_p = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=2)
frontera_salida_v = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=1)


# Modelo de la PINN en base a la ecuación de Navier-Stokes en 2D

datos = dde.data.PDE(geom,
    NavierStokes,
    [frontera_pared_u, frontera_pared_v, frontera_entrada_u, frontera_entrada_v, frontera_salida_p, frontera_salida_v],
    num_domain = 2600,
    num_boundary = 400,
    num_test = 10000
)

# Visualización de los puntos de entrenamiento
plt.figure(figsize=(5, 4))
plt.scatter(datos.train_x_all[:, 0], datos.train_x_all[:, 1], s=0.5)
plt.xlabel('Longitud a lo largo del eje x')
plt.ylabel('Distancia desde el eje medio del canal')
plt.show()


# Arquitectura de la red neuronal informada por la física

pinn = dde.maps.FNN([2] + 4 * [50] + [3], "tanh", "Glorot uniform")

# Creación y entrenamiento del modelo
modelo = dde.Model(datos, pinn)
modelo.compile("adam", lr=1e-3)

losshistory, train_state = modelo.train(iterations=20000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)



# Gráfica de resultados del modelo
samples = geom.random_points(500000)
resultado = modelo.predict(samples)
colores = [[0, 1.5], [-0.3, 0.3], [0, 35]]
titulos = ['Predicción para u', 'Predicción para v', 'Predicción para p']

for graf in range(3):
    plt.figure(figsize=(15, 4))
    plt.scatter(samples[:, 0], samples[:, 1], c=resultado[:, graf], cmap='jet', s=2)
    plt.colorbar()
    plt.clim(colores[graf])
    plt.title(titulos[graf])
    plt.tight_layout()
    plt.show()

# Geometría y condiciones de frontera el problema

rho = 1
mu = 1
u_in = 1.85
h = 1.95
L = 5


def frontera_pared(X, en_frontera):
    return np.logical_and(np.logical_or(np.isclose(X[1], -h / 2), np.isclose(X[1], h / 2)), en_frontera)

def frontera_entrada(X, en_frontera):
    return np.logical_and(np.isclose(X[0], -L / 2), en_frontera)

def frontera_salida(X, en_frontera):
    return np.logical_and(np.isclose(X[0], L / 2), en_frontera)

#Definición de la ecuación de Navier-Stokes en 2D
def NavierStokes(X, Y):
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)
    du_xx = dde.grad.hessian(Y, X, i=0, j=0, component=0)
    du_yy = dde.grad.hessian(Y, X, i=1, j=1, component=0)
    dv_xx = dde.grad.hessian(Y, X, i=0, j=0, component=1)
    dv_yy = dde.grad.hessian(Y, X, i=1, j=1, component=1)

    NavierStokes_u = Y[:, 0:1]*du_x + Y[:, 1:2]*du_y + 1/rho * dp_x - (mu/rho)*(du_xx + du_yy)
    NavierStokes_v = Y[:, 0:1]*dv_x + Y[:, 1:2]*dv_y + 1/rho * dp_y - (mu/rho)*(dv_xx + dv_yy)
    NavierStokes_cont = du_x + dv_y

    return [NavierStokes_u, NavierStokes_v, NavierStokes_cont]


# Geometría del problema

geom = dde.geometry.Rectangle(xmin=[-L / 2, -h / 2], xmax=[L / 2, h / 2])

# Condiciones de frontera

frontera_pared_u = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=0)
frontera_pared_v = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=1)

frontera_entrada_u = dde.DirichletBC(geom, lambda X: u_in, frontera_entrada, component=0)
frontera_entrada_v = dde.DirichletBC(geom, lambda X: 0., frontera_entrada, component=1)

frontera_salida_p = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=2)
frontera_salida_v = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=1)


# Modelo de la PINN en base a la ecuación de Navier-Stokes en 2D

datos = dde.data.PDE(geom,
    NavierStokes,
    [frontera_pared_u, frontera_pared_v, frontera_entrada_u, frontera_entrada_v, frontera_salida_p, frontera_salida_v],
    num_domain = 2600,
    num_boundary = 400,
    num_test = 10000
)

# Visualización de los puntos de entrenamiento
plt.figure(figsize=(5, 4))
plt.scatter(datos.train_x_all[:, 0], datos.train_x_all[:, 1], s=0.5)
plt.xlabel('Longitud a lo largo del eje x')
plt.ylabel('Distancia desde el eje medio del canal')
plt.show()


# Arquitectura de la red neuronal informada por la física

pinn = dde.maps.FNN([2] + 4 * [50] + [3], "tanh", "Glorot uniform")

# Creación y entrenamiento del modelo
modelo = dde.Model(datos, pinn)
modelo.compile("adam", lr=1e-3)

losshistory, train_state = modelo.train(iterations=25000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)



# Gráfica de resultados del modelo
samples = geom.random_points(500000)
resultado = modelo.predict(samples)
colores = [[0, 1.5], [-0.3, 0.3], [0, 35]]
titulos = ['Predicción para u', 'Predicción para v', 'Predicción para p']

for graf in range(3):
    plt.figure(figsize=(15, 4))
    plt.scatter(samples[:, 0], samples[:, 1], c=resultado[:, graf], cmap='jet', s=2)
    plt.colorbar()
    plt.clim(colores[graf])
    plt.title(titulos[graf])
    plt.tight_layout()
    plt.show()

# Geometría y condiciones de frontera el problema

rho = 1
mu = 1
u_in = 1.90
h = 1.95
L = 5


def frontera_pared(X, en_frontera):
    return np.logical_and(np.logical_or(np.isclose(X[1], -h / 2), np.isclose(X[1], h / 2)), en_frontera)

def frontera_entrada(X, en_frontera):
    return np.logical_and(np.isclose(X[0], -L / 2), en_frontera)

def frontera_salida(X, en_frontera):
    return np.logical_and(np.isclose(X[0], L / 2), en_frontera)

#Definición de la ecuación de Navier-Stokes en 2D
def NavierStokes(X, Y):
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)
    du_xx = dde.grad.hessian(Y, X, i=0, j=0, component=0)
    du_yy = dde.grad.hessian(Y, X, i=1, j=1, component=0)
    dv_xx = dde.grad.hessian(Y, X, i=0, j=0, component=1)
    dv_yy = dde.grad.hessian(Y, X, i=1, j=1, component=1)

    NavierStokes_u = Y[:, 0:1]*du_x + Y[:, 1:2]*du_y + 1/rho * dp_x - (mu/rho)*(du_xx + du_yy)
    NavierStokes_v = Y[:, 0:1]*dv_x + Y[:, 1:2]*dv_y + 1/rho * dp_y - (mu/rho)*(dv_xx + dv_yy)
    NavierStokes_cont = du_x + dv_y

    return [NavierStokes_u, NavierStokes_v, NavierStokes_cont]


# Geometría del problema

geom = dde.geometry.Rectangle(xmin=[-L / 2, -h / 2], xmax=[L / 2, h / 2])

# Condiciones de frontera

frontera_pared_u = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=0)
frontera_pared_v = dde.DirichletBC(geom, lambda X: 0., frontera_pared, component=1)

frontera_entrada_u = dde.DirichletBC(geom, lambda X: u_in, frontera_entrada, component=0)
frontera_entrada_v = dde.DirichletBC(geom, lambda X: 0., frontera_entrada, component=1)

frontera_salida_p = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=2)
frontera_salida_v = dde.DirichletBC(geom, lambda X: 0., frontera_salida, component=1)


# Modelo de la PINN en base a la ecuación de Navier-Stokes en 2D

datos = dde.data.PDE(geom,
    NavierStokes,
    [frontera_pared_u, frontera_pared_v, frontera_entrada_u, frontera_entrada_v, frontera_salida_p, frontera_salida_v],
    num_domain = 2600,
    num_boundary = 400,
    num_test = 10000
)

# Visualización de los puntos de entrenamiento
plt.figure(figsize=(5, 4))
plt.scatter(datos.train_x_all[:, 0], datos.train_x_all[:, 1], s=0.5)
plt.xlabel('Longitud a lo largo del eje x')
plt.ylabel('Distancia desde el eje medio del canal')
plt.show()


# Arquitectura de la red neuronal informada por la física

pinn = dde.maps.FNN([2] + 4 * [50] + [3], "tanh", "Glorot uniform")

# Creación y entrenamiento del modelo
modelo = dde.Model(datos, pinn)
modelo.compile("adam", lr=1e-3)

losshistory, train_state = modelo.train(iterations=25000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)



# Gráfica de resultados del modelo
samples = geom.random_points(500000)
resultado = modelo.predict(samples)
colores = [[0, 1.5], [-0.3, 0.3], [0, 35]]
titulos = ['Predicción para u', 'Predicción para v', 'Predicción para p']

for graf in range(3):
    plt.figure(figsize=(15, 4))
    plt.scatter(samples[:, 0], samples[:, 1], c=resultado[:, graf], cmap='jet', s=2)
    plt.colorbar()
    plt.clim(colores[graf])
    plt.title(titulos[graf])
    plt.tight_layout()
    plt.show()