import numpy as np
import cython
cimport numpy as cnp

cnp.import_array()

DTYPE = np.float64

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False) 
@cython.wraparound(False) 

def RungeKutta4(edo_f, cnp.ndarray y0, cnp.ndarray[DTYPE_t, ndim=1] tlist):
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

    cdef float dt = tlist[1] - tlist[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] y
    y = np.zeros([len(tlist), len(y0)], dtype=DTYPE)
    cdef int i
    cdef k1, k2, k3, k4

    y[0] = y0
    
    for i in range(0, len(tlist)-1):
        k1 = edo_f(tlist[i], y[i])
        k2 = edo_f(tlist[i] + dt/2, y[i] + dt*k1/2)
        k3 = edo_f(tlist[i] + dt/2, y[i] + dt*k2/2)
        k4 = edo_f(tlist[i] + dt, y[i] + dt*k3)
        y[i+1] = y[i]+ (k1 + 2*k2 + 2*k3 + k4)*dt/6
    return y