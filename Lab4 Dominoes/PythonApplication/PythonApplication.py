import taichi as ti
from taichi.math import *
ti.init(ti.gpu)

dt = 1e-3
vertices = ti.Vector.field(3, dtype = float, shape = 8)
indices = ti.field(ti.i32, shape = 36)

vertices[0] = (-0.05,-0.1,0.1);vertices[1] = (-0.05,-0.1,-0.1);vertices[2] = (0.05,-0.1,-0.1);vertices[3] = (0.05,-0.1,0.1)
vertices[4] = (-0.05,0.1,0.1);vertices[5] = (-0.05,0.1,-0.1);vertices[6] = (0.05,0.1,-0.1);vertices[7] = (0.05,0.1,0.1)

indices[0] = 0;indices[1] = 1;indices[2] = 2;indices[3] = 0;indices[4] = 2;indices[5] = 3;indices[6] = 0;indices[7] = 1;indices[8] = 4;
indices[9] = 1;indices[10] = 4;indices[11] = 5;indices[12] = 1;indices[13] = 2;indices[14] = 5;indices[15] = 2;indices[16] = 5;indices[17] = 6
indices[18] = 2;indices[19] = 3;indices[20] = 6;indices[21] = 3;indices[22] = 6;indices[23] = 7;indices[24] = 0;indices[25] = 3;indices[26] = 4
indices[27] = 3;indices[28] = 4;indices[29] = 7;indices[30] = 4;indices[31] = 5;indices[32] = 7;indices[33] = 5;indices[34] = 6;indices[35] = 7

p0 = ti.Vector.field(3, dtype = float, shape = 8)
p1 = ti.Vector.field(3, dtype = float, shape = 8)
p2 = ti.Vector.field(3, dtype = float, shape = 8)
p3 = ti.Vector.field(3, dtype = float, shape = 8)
p4 = ti.Vector.field(3, dtype = float, shape = 8)
p5 = ti.Vector.field(3, dtype = float, shape = 8)
p6 = ti.Vector.field(3, dtype = float, shape = 8)
p7 = ti.Vector.field(3, dtype = float, shape = 8)
p8 = ti.Vector.field(3, dtype = float, shape = 8)
p9 = ti.Vector.field(3, dtype = float, shape = 8)
p10 = ti.Vector.field(3, dtype = float, shape = 8)
p11 = ti.Vector.field(3, dtype = float, shape = 8)
p12 = ti.Vector.field(3, dtype = float, shape = 8)
w = ti.field(ti.f32, shape = 13)
theta = ti.field(ti.f32, shape = 13)

for j in range(8):
    p0[j] = vertices[j] + ti.Vector((-0.2, 0, - 0.2 - 0.1 * sqrt(3)))
    p1[j] = vertices[j] + ti.Vector((0, 0, - 0.2 - 0.1 * sqrt(3)))
    p2[j] = vertices[j] + ti.Vector((0.2, 0, - 0.2 - 0.1 * sqrt(3)))
    p3[j] = ti.Matrix([[cos(pi/3),0,-sin(pi/3)],[0,1,0],[sin(pi/3),0,cos(pi/3)]]) @ vertices[j] + ti.Vector((0.2 + 0.1 * sqrt(3), 0, - 0.1 - 0.1 * sqrt(3)))
    p4[j] = ti.Matrix([[cos(2*pi/3),0,-sin(2*pi/3)],[0,1,0],[sin(2*pi/3),0,cos(2*pi/3)]]) @ vertices[j] + ti.Vector((0.2 + 0.1 * sqrt(3), 0, - 0.1))
    p5[j] = ti.Matrix([[cos(pi),0,-sin(pi)],[0,1,0],[sin(pi),0,cos(pi)]]) @ vertices[j] + ti.Vector((0.2, 0, 0))
    p6[j] = ti.Matrix([[cos(pi),0,-sin(pi)],[0,1,0],[sin(pi),0,cos(pi)]]) @ vertices[j]
    p7[j] = vertices[j] + ti.Vector((-0.2, 0, 0))
    p8[j] = ti.Matrix([[cos(2*pi/3),0,-sin(2*pi/3)],[0,1,0],[sin(2*pi/3),0,cos(2*pi/3)]]) @ vertices[j] + ti.Vector((- 0.2 - 0.1 * sqrt(3), 0, 0.1))
    p9[j] = ti.Matrix([[cos(pi/3),0,-sin(pi/3)],[0,1,0],[sin(pi/3),0,cos(pi/3)]]) @ vertices[j] + ti.Vector((- 0.2 - 0.1 * sqrt(3), 0, 0.1 + 0.1 * sqrt(3)))
    p10[j] = vertices[j] + ti.Vector((- 0.2, 0, 0.2 + 0.1 * sqrt(3)))
    p11[j] = vertices[j] + ti.Vector((0, 0, 0.2 + 0.1 * sqrt(3)))
    p12[j] = vertices[j] + ti.Vector((0.2, 0, 0.2 + 0.1 * sqrt(3)))

window = ti.ui.Window("", (648, 648), vsync = True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0, 3, 3)
camera.lookat(0, 0, 0)
scene.set_camera(camera)

@ti.func
def rotate(x:vec3, o:vec3, y:vec3, theta:ti.f32) -> vec3:
    left = (x - o).normalized()
    up = -(x - o).cross(y - o).normalized()
    return o + (x - o).norm() * sin(theta) * up + (x - o).norm() * cos(theta) * left

@ti.kernel
def update():

    p0[0] = rotate(p0[0],p0[3],p0[2],w[0])
    p0[1] = p0[0] + p0[2] - p0[3]
    p0[4] = rotate(p0[4],p0[3],p0[2],w[0])
    p0[5] = p0[4] + p0[2] - p0[3]
    p0[7] = rotate(p0[7],p0[3],p0[2],w[0])
    p0[6] = p0[7] + p0[2] - p0[3]

    w[0] = (p0[3] - p0[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()

while window.running:
    scene.ambient_light((0.1, 0.1, 0.1))
    scene.point_light(pos = (0, 3, -3), color = (1, 1, 1))

    w[0] = 0.01
    update()

    scene.mesh(p0, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p1, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p2, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p3, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p4, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p5, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p6, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p7, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p8, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p9, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p10, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p11, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    scene.mesh(p12, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    canvas.scene(scene)
    window.show()