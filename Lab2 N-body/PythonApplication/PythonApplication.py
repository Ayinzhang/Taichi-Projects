import taichi as ti
ti.init(ti.gpu)

n = 300
G = 6.67e-11
block_size = 12
galaxy_size = 0.45
planet_radius = 2
dt = 1e-5
substepping = 10

r = ti.field(ti.f32, n)
m = ti.field(ti.f32, n)
p = ti.Vector.field(2, ti.f32, n)
v = ti.Vector.field(2, ti.f32, n)
f = ti.Vector.field(2, ti.f32, n)
b = ti.field(ti.i32)
block = ti.root.dense(ti.ij, [block_size, block_size])
index = block.dynamic(ti.k, n)
index.place(b)

@ti.kernel
def init():
    center = ti.Vector([0.5, 0.5])
    for i in range(n):
        theta = ti.random() * 2 * ti.math.pi
        d = (ti.sqrt(ti.random()) * 0.7 + 0.3) * galaxy_size
        offset = d * ti.Vector([ti.cos(theta), ti.sin(theta)])
        r[i] = (ti.random() + 0.5) * planet_radius
        m[i] = r[i] ** 3;
        p[i] = center + offset
        b[ti.math.floor(p[i].x * block_size, ti.i32), ti.math.floor(p[i].y * block_size, ti.i32)].append(i)
        v[i] = [-offset.y, offset.x]
        v[i] *= 120 * (ti.sqrt(ti.random()) + 0.5)

@ti.kernel
def compute():
   for i in range(n):
       f[i] = ti.Vector([0.0, 0.0])
   for i in range(n):
       x = ti.math.clamp(ti.math.floor(p[i].x * block_size, ti.i32), 0, block_size - 1)
       y = ti.math.clamp(ti.math.floor(p[i].y * block_size, ti.i32), 0, block_size - 1)
       for j in range(-1, 2):
           for k in range(-1, 2):
               if 0 <= x + j and x + j < block_size and 0 <= y + k and y + k < block_size:
                   for l in range(b[x + j, y + k].length()):
                       o = b[x + j, y + k, l]
                       if i != o:
                           r = ti.math.length(p[i] - p[o])
                           force = G * m[i] * m[o] / (r ** 3) * (p[o] - p[i])
                           f[i] += force
                           f[o] -= force

@ti.kernel
def pull(xpos:ti.f32, ypos:ti.f32):
    pos = [xpos, ypos]
    for i in range(n):
        f[i] += (pos - p[i]) ** 3 * 12000000

@ti.kernel
def push(xpos:ti.f32, ypos:ti.f32):
    pos = [xpos, ypos]
    for i in range(n):
        f[i] += (p[i] - pos) ** 3 * 12000000

@ti.kernel
def update():
   for i in range(block_size):
       for j in range(block_size):
           b[i, j].deactivate()
   for i in range(n):
       v[i] += dt * f[i] / m[i]
       p[i] += dt * v[i]
       x = ti.math.clamp(ti.math.floor(p[i].x * block_size, ti.i32), 0, block_size - 1)
       y = ti.math.clamp(ti.math.floor(p[i].y * block_size, ti.i32), 0, block_size - 1)
       b[x, y].append(i)
    

gui = ti.GUI("n-body", (512, 512))
init()
while gui.running:
    compute()
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.LMB:
            pull(e.pos[0], e.pos[1])
        if e.key == ti.GUI.RMB:
            push(e.pos[0], e.pos[1])
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
    update()
    gui.clear(0x112F41)
    gui.circles(p.to_numpy(), color=0xffffff, radius=r.to_numpy())
    gui.show()