import taichi as ti
from taichi.math import *
ti.init(ti.gpu)

n = 24
dt = 1e-5
water_size = 0.3
quad_size = 2 * water_size / n
box_size = 0.5
block_size = 10

g = ti.Vector([0, -9.8, 0])
p = ti.Vector.field(3, dtype = float, shape = n * n * n)
v = ti.Vector.field(3, dtype = float, shape = n * n * n)
index = ti.field(ti.i32)
block = ti.root.dense(ti.i, block_size * block_size * block_size)
data = block.dynamic(ti.j, n * n * n)
data.place(index)

for i,j,k in ti.ndrange(n, n, n):
    p[i + j * n + k * n * n] = ti.Vector([-water_size + i * quad_size, -water_size + j * quad_size, -water_size + k * quad_size])

window = ti.ui.Window("Water-simulation", (648, 648), vsync = True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0, 1.5, 1.5)
camera.lookat(0, 0, 0)
scene.set_camera(camera)

@ti.kernel
def update():
    for i in range(block_size * block_size * block_size):
        index[i].deactivate()
    for i,j,k in ti.ndrange(n, n, n):
        v[i + j * n + k * n * n] += g * dt
        p[i + j * n + k * n * n] += v[i + j * n + k * n * n]
        for o in range(3):
            if p[i + j * n + k * n * n][o] < -box_size:
                v[i + j * n + k * n * n][o] = 0
                p[i + j * n + k * n * n][o] = -box_size
            elif p[i + j * n + k * n * n][o] > box_size:
                v[i + j * n + k * n * n][o] = 0
                p[i + j * n + k * n * n][o] = box_size
        index[int((box_size + p[i + j * n + k * n * n][0])//(2 * box_size / block_size)) +
              int((box_size + p[i + j * n + k * n * n][1])//(2 * box_size / block_size)) * block_size +
              int((box_size + p[i + j * n + k * n * n][2])//(2 * box_size / block_size)) * block_size * block_size].append(i + j * n + k * n * n)
    for i in range(block_size * block_size * block_size):
        for j in range(index[i].length()):
            for k in range(j):
                if (p[index[i, j]] - p[index[i, k]]).norm() < 2 * quad_size:
                    pjk = (p[index[i, j]] - p[index[i, k]]).normalized()
                    offset = (2 * quad_size - (p[index[i, j]] - p[index[i, k]]).norm()) / 2
                    v[index[i, j]] = v[index[i, j]] + v[index[i, j]].dot(pjk) * pjk + pjk * v[index[i, j]].dot(pjk)
                    v[index[i, k]] = v[index[i, k]] - v[index[i, k]].dot(pjk) * pjk - pjk * v[index[i, k]].dot(pjk)
                    p[index[i, j]] = p[index[i, j]] + offset * pjk
                    p[index[i, k]] = p[index[i, k]] - offset * pjk


while window.running:
    update()
    #scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.particles(p, radius = quad_size, color = (13/255, 144/255, 191/255))
    canvas.scene(scene)
    window.show()