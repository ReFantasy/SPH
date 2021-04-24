import taichi as ti
from helper import *
import Particles
from kernel_func import w


@ti.kernel
def init():
    """
    初始化质量
    """
    for i in range(Particles.N):
        Particles.m[i] = 1.0


@ti.kernel
def simulation_loop():
    """
    计算密度和压力
    """
    for i in range(Particles.N):
        for j in range(Particles.N):
            Particles.density[i] += (Particles.m[j] * w(i, j, Particles.h))

    for i in range(Particles.N):
        Particles.p[i] = compute_p(Particles.density[i])

    """
    计算受力
    """
    for i in range(Particles.N):
        Particles.f_pressure[i] = -Particles.m[i] / Particles.density[i] * grad_p(i)
        print(Particles.f_pressure[i])


if __name__ == "__main__":
    init()
    simulation_loop()
