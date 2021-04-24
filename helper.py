import taichi as ti
from kernel_func import dw_dxi, w
import Particles


@ti.func
def compute_p(density):
    return Particles.k * (ti.pow(density / Particles.po, 7) - 1)


@ti.func
def grad_p(i):
    pi = Particles.p[i]
    p = dw_dxi(0, 0, 1)*0
    for j in range(Particles.N):
       p+= Particles.density[i] * Particles.m[j] * (pi / ti.pow(Particles.density[i], 2) + Particles.p[j] / ti.pow(Particles.density[j], 2)) * dw_dxi(i, j, Particles.h)
    return p
