import taichi as ti
from taichi.math import *
ti.init(ti.gpu)

dt = 1e-2
vertices = ti.Vector.field(3, dtype = float, shape = 8)
indices = ti.field(ti.i32, shape = 13 * 36)

vertices[0] = (-0.05,-0.1,0.1);vertices[1] = (-0.05,-0.1,-0.1);vertices[2] = (0.05,-0.1,-0.1);vertices[3] = (0.05,-0.1,0.1)
vertices[4] = (-0.05,0.1,0.1);vertices[5] = (-0.05,0.1,-0.1);vertices[6] = (0.05,0.1,-0.1);vertices[7] = (0.05,0.1,0.1)

indices[0] = 0;indices[1] = 1;indices[2] = 2;indices[3] = 0;indices[4] = 2;indices[5] = 3;indices[6] = 0;indices[7] = 1;indices[8] = 4;
indices[9] = 1;indices[10] = 4;indices[11] = 5;indices[12] = 1;indices[13] = 2;indices[14] = 5;indices[15] = 2;indices[16] = 5;indices[17] = 6
indices[18] = 2;indices[19] = 3;indices[20] = 6;indices[21] = 3;indices[22] = 6;indices[23] = 7;indices[24] = 0;indices[25] = 3;indices[26] = 4
indices[27] = 3;indices[28] = 4;indices[29] = 7;indices[30] = 4;indices[31] = 5;indices[32] = 7;indices[33] = 5;indices[34] = 6;indices[35] = 7

p = ti.Vector.field(3, dtype = float, shape = 8 * 13)
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

w[0] = 1e-3

@ti.kernel
def init():
    for i in range(36, 13 * 36):
        indices[i] = indices[i % 36] + i // 36 * 8

    for j in range(8):
        p[j] = p0[j] = vertices[j] + ti.Vector((-0.2, 0, - 0.2 - 0.1 * sqrt(3)))
        p[j + 8] = p1[j] = vertices[j] + ti.Vector((0, 0, - 0.2 - 0.1 * sqrt(3)))
        p[j + 16] = p2[j] = vertices[j] + ti.Vector((0.2, 0, - 0.2 - 0.1 * sqrt(3)))
        p[j + 24] = p3[j] = ti.Matrix([[cos(pi/3),0,-sin(pi/3)],[0,1,0],[sin(pi/3),0,cos(pi/3)]]) @ vertices[j] + ti.Vector((0.2 + 0.1 * sqrt(3), 0, - 0.1 - 0.1 * sqrt(3)))
        p[j + 32] = p4[j] = ti.Matrix([[cos(2*pi/3),0,-sin(2*pi/3)],[0,1,0],[sin(2*pi/3),0,cos(2*pi/3)]]) @ vertices[j] + ti.Vector((0.2 + 0.1 * sqrt(3), 0, - 0.1))
        p[j + 40] = p5[j] = ti.Matrix([[cos(pi),0,-sin(pi)],[0,1,0],[sin(pi),0,cos(pi)]]) @ vertices[j] + ti.Vector((0.2, 0, 0))
        p[j + 48] = p6[j] = ti.Matrix([[cos(pi),0,-sin(pi)],[0,1,0],[sin(pi),0,cos(pi)]]) @ vertices[j]
        p[j + 56] = p7[j] = ti.Matrix([[cos(pi),0,-sin(pi)],[0,1,0],[sin(pi),0,cos(pi)]]) @ vertices[j] + ti.Vector((-0.2, 0, 0))
        p[j + 64] = p8[j] = ti.Matrix([[cos(2*pi/3),0,-sin(2*pi/3)],[0,1,0],[sin(2*pi/3),0,cos(2*pi/3)]]) @ vertices[j] + ti.Vector((- 0.2 - 0.1 * sqrt(3), 0, 0.1))
        p[j + 72] = p9[j] = ti.Matrix([[cos(pi/3),0,-sin(pi/3)],[0,1,0],[sin(pi/3),0,cos(pi/3)]]) @ vertices[j] + ti.Vector((- 0.2 - 0.1 * sqrt(3), 0, 0.1 + 0.1 * sqrt(3)))
        p[j + 80] = p10[j] = vertices[j] + ti.Vector((- 0.2, 0, 0.2 + 0.1 * sqrt(3)))
        p[j + 88] = p11[j] = vertices[j] + ti.Vector((0, 0, 0.2 + 0.1 * sqrt(3)))
        p[j + 96] = p12[j] = vertices[j] + ti.Vector((0.2, 0, 0.2 + 0.1 * sqrt(3)))

@ti.func
def rotate(x:vec3, o:vec3, y:vec3, theta:ti.f32) -> vec3:
    left = (x - o).normalized()
    up = -(x - o).cross(y - o).normalized()
    return o + (x - o).norm() * sin(theta * dt) * up + (x - o).norm() * cos(theta * dt) * left

@ti.kernel
def update():
    p0[0] = rotate(p0[0],p0[3],p0[2],w[0]);p0[1] = p0[0] + p0[2] - p0[3];p0[4] = rotate(p0[4],p0[3],p0[2],w[0])
    p0[5] = p0[4] + p0[2] - p0[3];p0[7] = rotate(p0[7],p0[3],p0[2],w[0]);p0[6] = p0[7] + p0[2] - p0[3]

    p1[0] = rotate(p1[0],p1[3],p1[2],w[1]);p1[1] = p1[0] + p1[2] - p1[3];p1[4] = rotate(p1[4],p1[3],p1[2],w[1])
    p1[5] = p1[4] + p1[2] - p1[3];p1[7] = rotate(p1[7],p1[3],p1[2],w[1]);p1[6] = p1[7] + p1[2] - p1[3]

    p2[0] = rotate(p2[0],p2[3],p2[2],w[2]);p2[1] = p2[0] + p2[2] - p2[3];p2[4] = rotate(p2[4],p2[3],p2[2],w[2])
    p2[5] = p2[4] + p2[2] - p2[3];p2[7] = rotate(p2[7],p2[3],p2[2],w[2]);p2[6] = p2[7] + p2[2] - p2[3]

    p3[0] = rotate(p3[0],p3[3],p3[2],w[3]);p3[1] = p3[0] + p3[2] - p3[3];p3[4] = rotate(p3[4],p3[3],p3[2],w[3])
    p3[5] = p3[4] + p3[2] - p3[3];p3[7] = rotate(p3[7],p3[3],p3[2],w[3]);p3[6] = p3[7] + p3[2] - p3[3]

    p4[0] = rotate(p4[0],p4[3],p4[2],w[4]);p4[1] = p4[0] + p4[2] - p4[3];p4[4] = rotate(p4[4],p4[3],p4[2],w[4])
    p4[5] = p4[4] + p4[2] - p4[3];p4[7] = rotate(p4[7],p4[3],p4[2],w[4]);p4[6] = p4[7] + p4[2] - p4[3]

    p5[0] = rotate(p5[0],p5[3],p5[2],w[5]);p5[1] = p5[0] + p5[2] - p5[3];p5[4] = rotate(p5[4],p5[3],p5[2],w[5])
    p5[5] = p5[4] + p5[2] - p5[3];p5[7] = rotate(p5[7],p5[3],p5[2],w[5]);p5[6] = p5[7] + p5[2] - p5[3]

    p6[0] = rotate(p6[0],p6[3],p6[2],w[6]);p6[1] = p6[0] + p6[2] - p6[3];p6[4] = rotate(p6[4],p6[3],p6[2],w[6])
    p6[5] = p6[4] + p6[2] - p6[3];p6[7] = rotate(p6[7],p6[3],p6[2],w[6]);p6[6] = p6[7] + p6[2] - p6[3]

    p7[0] = rotate(p7[0],p7[3],p7[2],w[7]);p7[1] = p7[0] + p7[2] - p7[3];p7[4] = rotate(p7[4],p7[3],p7[2],w[7])
    p7[5] = p7[4] + p7[2] - p7[3];p7[7] = rotate(p7[7],p7[3],p7[2],w[7]);p7[6] = p7[7] + p7[2] - p7[3]

    p8[0] = rotate(p8[0],p8[3],p8[2],w[8]);p8[1] = p8[0] + p8[2] - p8[3];p8[4] = rotate(p8[4],p8[3],p8[2],w[8])
    p8[5] = p8[4] + p8[2] - p8[3];p8[7] = rotate(p8[7],p8[3],p8[2],w[8]);p8[6] = p8[7] + p8[2] - p8[3]

    p9[0] = rotate(p9[0],p9[3],p9[2],w[9]);p9[1] = p9[0] + p9[2] - p9[3];p9[4] = rotate(p9[4],p9[3],p9[2],w[9])
    p9[5] = p9[4] + p9[2] - p9[3];p9[7] = rotate(p9[7],p9[3],p9[2],w[9]);p9[6] = p9[7] + p9[2] - p9[3]

    p10[0] = rotate(p10[0],p10[3],p10[2],w[10]);p10[1] = p10[0] + p10[2] - p10[3];p10[4] = rotate(p10[4],p10[3],p10[2],w[10])
    p10[5] = p10[4] + p10[2] - p10[3];p10[7] = rotate(p10[7],p10[3],p10[2],w[10]);p10[6] = p10[7] + p10[2] - p10[3]

    p11[0] = rotate(p11[0],p11[3],p11[2],w[11]);p11[1] = p11[0] + p11[2] - p11[3];p11[4] = rotate(p11[4],p11[3],p11[2],w[11])
    p11[5] = p11[4] + p11[2] - p11[3];p11[7] = rotate(p11[7],p11[3],p11[2],w[11]);p11[6] = p11[7] + p11[2] - p11[3]

    p12[0] = rotate(p12[0],p12[3],p12[2],w[12]);p12[1] = p12[0] + p12[2] - p12[3];p12[4] = rotate(p12[4],p12[3],p12[2],w[12])
    p12[5] = p12[4] + p12[2] - p12[3];p12[7] = rotate(p12[7],p12[3],p12[2],w[12]);p12[6] = p12[7] + p12[2] - p12[3]

    w[0] = w[0] + (p0[3] - p0[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[1] = w[1] + (p1[3] - p1[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[2] = w[2] + (p2[3] - p2[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[3] = w[3] + (p3[3] - p3[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[4] = w[4] + (p4[3] - p4[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[5] = w[5] + (p5[3] - p5[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[6] = w[6] + (p6[3] - p6[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[7] = w[7] + (p7[3] - p7[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[8] = w[8] + (p8[3] - p8[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[9] = w[9] + (p9[3] - p9[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[10] = w[10] + (p10[3] - p10[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[11] = w[11] + (p11[3] - p11[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()
    w[12] = w[12] + (p12[3] - p12[7]).normalized().cross(ti.Vector((0, -1, 0))).norm()

    if (p1[5] - p1[0]).cross(p1[1] - p1[0]).dot(rotate(p0[7],p0[3],p0[2],w[0]) - p1[0]) < 0 or (p1[5] - p1[0]).cross(p1[1] - p1[0]).dot(p0[3] - p1[0]) < 0:
        w[1] = w[1] + w[0] * (p0[1] - p0[0]).normalized().dot((p1[1] - p1[0]).normalized())
        w[0] = 0

    if (p2[5] - p2[0]).cross(p2[1] - p2[0]).dot(rotate(p1[7],p1[3],p1[2],w[1]) - p2[0]) < 0 or (p2[5] - p2[0]).cross(p2[1] - p2[0]).dot(p1[3] - p2[0]) < 0:
        w[2] = w[2] + w[1] * (p1[1] - p1[0]).normalized().dot((p2[1] - p2[0]).normalized())
        w[1] = 0

    if (p3[5] - p3[0]).cross(p3[1] - p3[0]).dot(rotate(p2[7],p2[3],p2[2],w[2]) - p3[0]) < 0 or (p3[5] - p3[0]).cross(p2[1] - p2[0]).dot(p1[3] - p2[0]) < 0:
        w[3] = w[3] + w[2] * (p2[1] - p2[0]).normalized().dot((p3[1] - p3[0]).normalized())
        w[2] = 0

    if (p4[5] - p4[0]).cross(p4[1] - p4[0]).dot(rotate(p3[7],p3[3],p3[2],w[3]) - p4[0]) < 0 or (p4[5] - p4[0]).cross(p4[1] - p4[0]).dot(p3[3] - p4[0]) < 0:
        w[4] = w[4] + w[3] * (p3[1] - p3[0]).normalized().dot((p4[1] - p4[0]).normalized())
        w[3] = 0

    if (p5[5] - p5[0]).cross(p5[1] - p5[0]).dot(rotate(p4[7],p4[3],p4[2],w[4]) - p5[0]) < 0 or (p5[5] - p5[0]).cross(p5[1] - p5[0]).dot(p4[3] - p5[0]) < 0:
        w[5] = w[5] + w[4] * (p4[1] - p4[0]).normalized().dot((p5[1] - p5[0]).normalized())
        w[4] = 0

    if (p6[5] - p6[0]).cross(p6[1] - p6[0]).dot(rotate(p5[7],p5[3],p5[2],w[5]) - p6[0]) < 0 or (p6[5] - p6[0]).cross(p6[1] - p6[0]).dot(p5[3] - p6[0]) < 0:
        w[6] = w[6] + w[5] * (p5[1] - p5[0]).normalized().dot((p6[1] - p6[0]).normalized())
        w[5] = 0

    if (p7[5] - p7[0]).cross(p7[1] - p7[0]).dot(rotate(p6[7],p6[3],p6[2],w[6]) - p7[0]) < 0 or (p7[5] - p7[0]).cross(p7[1] - p7[0]).dot(p6[3] - p7[0]) < 0:
        w[7] = w[7] + w[6] * (p6[1] - p6[0]).normalized().dot((p7[1] - p7[0]).normalized())
        w[6] = 0

    if (p8[5] - p8[0]).cross(p8[1] - p8[0]).dot(rotate(p7[7],p7[3],p7[2],w[7]) + p7[2] - p7[3] - p8[0]) < 0 or (p8[5] - p8[0]).cross(p8[1] - p8[0]).dot(p7[2] - p8[0]) < 0:
        w[8] = w[8] + w[7] * (p7[1] - p7[0]).normalized().dot((p8[1] - p8[0]).normalized())
        w[7] = 0

    if (p9[5] - p9[0]).cross(p9[1] - p9[0]).dot(rotate(p8[7],p8[3],p8[2],w[8]) + p8[2] - p8[3] - p9[0]) < 0 or (p9[5] - p9[0]).cross(p9[1] - p9[0]).dot(p8[2] - p9[0]) < 0:
        w[9] = w[9] + w[8] * (p8[1] - p8[0]).normalized().dot((p9[1] - p9[0]).normalized())
        w[8] = 0

    if (p10[5] - p10[0]).cross(p10[1] - p10[0]).dot(rotate(p9[7],p9[3],p9[2],w[9]) + p9[2] - p9[3] - p10[0]) < 0 or (p10[5] - p10[0]).cross(p10[1] - p10[0]).dot(p9[2] - p10[0]) < 0:
        w[10] = w[10] + w[9] * (p9[1] - p9[0]).normalized().dot((p10[1] - p10[0]).normalized())
        w[9] = 0

    if (p11[5] - p11[0]).cross(p11[1] - p11[0]).dot(rotate(p10[7],p10[3],p10[2],w[10]) - p11[0]) < 0 or (p11[5] - p11[0]).cross(p11[1] - p11[0]).dot(p10[3] - p11[0]) < 0:
        w[11] = w[11] + w[10] * (p10[1] - p10[0]).normalized().dot((p11[1] - p11[0]).normalized())
        w[10] = 0

    if (p12[5] - p12[0]).cross(p12[1] - p12[0]).dot(rotate(p11[7],p11[3],p11[2],w[11]) - p12[0]) < 0 or (p12[5] - p12[0]).cross(p12[1] - p12[0]).dot(p11[3] - p12[0]) < 0:
        w[12] = w[12] + w[11] * (p11[1] - p11[0]).normalized().dot((p12[1] - p12[0]).normalized())
        w[11] = 0

    if rotate(p12[7],p12[3],p12[2],w[12]).y < 0:
        w[12] = 0

    for i in range(8):
        p[i] = p0[i]
        p[8 + i] = p1[i]
        p[16 + i] = p2[i]
        p[24 + i] = p3[i]
        p[32 + i] = p4[i]
        p[40 + i] = p5[i]
        p[48 + i] = p6[i]
        p[56 + i] = p7[i]
        p[64 + i] = p8[i]
        p[72 + i] = p9[i]
        p[80 + i] = p10[i]
        p[88 + i] = p11[i]
        p[96 + i] = p12[i]

window = ti.ui.Window("Dominoes", (648, 648), vsync = True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0, 3, 3)
camera.lookat(0, 0, 0)
scene.set_camera(camera)
init()
while window.running:
    scene.ambient_light((0.1, 0.1, 0.1))
    scene.point_light(pos = (0, 3, -3), color = (1, 1, 1))

    update()
    #Combine Mesh
    scene.mesh(p, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p0, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p1, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p2, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p3, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p4, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p5, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p6, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p7, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p8, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p9, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p10, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p11, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    #scene.mesh(p12, indices = indices, color = (1, 0.89, 0.8), two_sided = True)
    canvas.scene(scene)
    window.show()