import taichi as ti
import math
import Particles


@ti.func
def w(i, j, h) -> ti.f32:
    diff_x = Particles.X[i] - Particles.X[j]
    norm = diff_x.norm()
    q = norm / h
    d = 3

    if 0 <= q < 1:
        q = 2 / 3 - ti.pow(q, 2) + 1 / 2 * ti.pow(q, 3)
    elif 1 <= q < 2:
        q = 1 / 6 * ti.pow(2 - q, 3)
    else:
        q = 0

    q *= (3 / 2 / math.pi)

    return 1 / ti.pow(h, d) * q


@ti.func
def dw_dxi(i, j, h):
    diff_x = Particles.X[i] - Particles.X[j]
    norm = diff_x.norm()
    q = norm / h

    dx = 1 / h / norm * diff_x

    c = 1 / ti.pow(h, 3) * 3 / 2 / math.pi

    res = 0 * dx
    if 0 <= q < 1:
        res = c * (-2 * q + 3 / 2 * q * q) * dx
    elif 1 <= q < 2:
        res = -c * 0.5 * (2 - q) * (2 - q) * dx
    else:
        res = 0 * dx

    return res

# if __name__ == '__main__':
# Particles.X[0] = ti.Vector([2, 4, 0])
# Particles.X[1] = ti.Vector([2.5, 3, 1])
#
#
# @ti.kernel
# def test():
#     res = dw_dxi(0, 1, 5)
#     print(res)
#     Particles.z[None] = w(1, 0, 5)
#
#
# with ti.Tape(Particles.z):
#     test()
#
# print('dz/dx =', Particles.X.grad[0][0], Particles.X.grad[0][1], Particles.X.grad[0][2])
