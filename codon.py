from python import numpy as np
from python import timeit

def RungeKutta4(edo_f, y0, tlist):
    '''
    Resolve a EDO dy/dt = edo(t, y) via método de Runge Kutta de 4° ordem.
    Considerando a condição inicial y(t0) = y0,
    sendo t0=tlist[0], com passos temporais dt=tlist[1]-tlist[0]
    para todos os instantes de tempo t em tlist.

    Neste código assumimos que y é um vetor cujas componentesse referem às incógnitas da EDO.

    Entrada
    -------
    edo_f: função de (t,y) representando o lado direito da EDO
    y0: (array) com os valores das condições iniciais
    tlist: lista de instantes de tempo

    Saída
    -----
    Matriz (array) com a solução y(t). Cada linha da matriz
    se refere a um instante de tempo de tlist, e as colunas
    representam as componentes do vetor y.
    '''

    dt = tlist[1] - tlist[0]
    y = np.zeros([len(tlist), len(y0)])
    y[0] = y0
    for i in range(len(tlist)-1):
        k1 = edo_f(tlist[i], y[i])
        k2 = edo_f(tlist[i] + dt/2, y[i] + dt*k1/2)
        k3 = edo_f(tlist[i] + dt/2, y[i] + dt*k2/2)
        k4 = edo_f(tlist[i] + dt, y[i] + dt*k3)
        y[i+1] = y[i]+ (k1 + 2*k2 + 2*k3 + k4)*dt/6
    return y

alpha = 1.5
beta = 0.4
delta = 0.1
gamma = 0.4

def lotkavolterra(t, y):
    x = y[0]
    z = y[1]
    f = np.array([alpha*x - beta*x*z, delta*x*z - gamma*z])  # f[dx/dt, dz/dt]
    return f

time = time = [np.linspace(0, 100, 1000), np.linspace(0, 100, 5000), np.linspace(0, 100, 10000), np.linspace(0, 100, 20000)]
y0 = np.array([10, 10])

starttime0 = timeit.default_timer()
RungeKutta4(lotkavolterra, y0, time[0])
t0 = timeit.default_timer() - starttime0

starttime1 = timeit.default_timer()
RungeKutta4(lotkavolterra, y0, time[1])
t1 = timeit.default_timer() - starttime1

starttime2 = timeit.default_timer()
RungeKutta4(lotkavolterra, y0, time[2])
t2 = timeit.default_timer() - starttime2

starttime3 = timeit.default_timer()
RungeKutta4(lotkavolterra, y0, time[3])
t3 = timeit.default_timer() - starttime3

np.savez('time_codon.npz', t0=t0, t1=t1,t2=t2, t3=t3)