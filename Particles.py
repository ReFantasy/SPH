import taichi as ti
ti.init(arch=ti.cuda)
N = 15000

po = 10
k = 10
h = 0.1

'''
position of particles
'''
X = ti.Vector.field(3, dtype=ti.f32, shape=N)  # , needs_grad=True

'''
mass of particles
'''
m = ti.field(dtype=ti.f32, shape=N)

'''
density of particles
'''
density = ti.field(dtype=ti.f32, shape=N)

'''
pressure of particles
'''
p = ti.field(dtype=ti.f32, shape=N)

z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

f_pressure = ti.Vector.field(3, dtype=ti.f32, shape=N)