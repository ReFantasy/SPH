import taichi as ti
import math

ti.init(arch=ti.gpu)

gui = ti.GUI(name=str("TaichiGUI"), res=(600, 600), background_color=0x00456B)

N = 1000
x = ti.Vector.field(2, dtype=ti.f32, shape=N)  # , needs_grad=True
mass = ti.field(dtype=ti.f32, shape=N)
density = ti.field(dtype=ti.f32, shape=N)
pressure = ti.field(dtype=ti.f32, shape=N)
f_pressure = ti.Vector.field(2, dtype=ti.f32, shape=N)
v = ti.Vector.field(2, dtype=ti.f32, shape=N)
h = ti.field(ti.f32, shape=())
d = ti.field(ti.f32, shape=())
po = ti.field(ti.f32, shape=())
k = ti.field(ti.f32, shape=())

h[None] = 1.2
d[None] = 3
po[None] = 1
k[None] = 1


@ti.func
def w(i, j) -> ti.f32:
    vx_ij = x[i] - x[j]
    q = vx_ij.norm() / h

    if q < 1:
        q = 2 / 3 - ti.pow(q, 2) + 1 / 2 * ti.pow(q, 3)
    elif 1 <= q < 2:
        q = 1 / 6 * ti.pow(2 - q, 3)
    else:
        q = 0

    q *= (3 / (2 * math.pi))
    return 1 / ti.pow(h, d) * q


@ti.func
def dw(i, j, res=ti.Vector([0.0, 0.0])):
    vx_ij = x[i] - x[j]
    q = vx_ij.norm() / h
    alpha_x = 1 / h / vx_ij.norm() * vx_ij
    c = 1 / ti.pow(h, d) * 3 / 2 / math.pi

    if 0 <= q < 1:
        res = c * (-2 * q + 3 / 2 * q * q) * alpha_x
    elif 1 <= q < 2:
        res = -c * 0.5 * (2 - q) * (2 - q) * alpha_x
    else:
        res = 0 * alpha_x
    return res


@ti.func
def compute_pressure(den):
    return k * (ti.pow(den / po, 7) - 1)


@ti.func
def dp(i, res=ti.Vector([0.0, 0.0])):
    for j in range(N):
        if j != i:
            res += mass[j] * (pressure[i] / ti.pow(density[i], 2) + pressure[j] / ti.pow(density[j], 2)) * dw(i, j)
    return density[i] * res


@ti.kernel
def simulation_loop():
    for i in range(N):
        density[i] = 0
        for j in range(N):
            density[i] += mass[j] * w(i, j)

    for i in range(N):
        pressure[i] = compute_pressure(density[i])

    for i in range(N):
        f_pressure[i] = -mass[i] / density[i] * dp(i)

    for i in range(N):
        g = ti.Vector([0, -9.8])
        v[i] = v[i] + 0.01 * (f_pressure[i] + g) / mass[i]
        x[i] = x[i] + 0.01 * v[i]


if __name__ == "__main__":
    for i in range(N):
        x[i] = ti.Vector([i % 10, i / 10])
        # print(i % 10, i / 10)
        mass[i] = 2.0
        density[i] = 3.0
        pressure[i] = 4.0

    while True:
        if gui.get_event(ti.GUI.ESCAPE):
            break
        simulation_loop()
        for i in range(N):
            pos = x[i]
            gui.circle(pos=(pos[0] / 10, pos[1] / 10), color=0xFF0000, radius=5)
        gui.show()
