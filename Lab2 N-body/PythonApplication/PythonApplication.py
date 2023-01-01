import taichi as ti
ti.init(ti.gpu)

n = 300
G = 6.67e-11
galaxy_size = 0.45
planet_radius = 2
dt = 1e-5
substepping = 10

r = ti.field(dtype = ti.f32, shape = n)
m = ti.field(dtype = ti.f32, shape = n)
p = ti.Vector.field(2, ti.f32, n)
v = ti.Vector.field(2, ti.f32, n)
f = ti.Vector.field(2, ti.f32, n)

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
        v[i] = [-offset.y, offset.x]
        v[i] *= 120 * (ti.sqrt(ti.random()) + 0.5)

@ti.kernel
def compute():
   for i in range(n):
       f[i] = ti.Vector([0.0, 0.0])
   for i in range(n):
       for j in range(i + 1, n):
           r = ti.math.length(p[i] - p[j])
           force = G * m[i] * m[j] * (r ** 3) * (p[j] - p[i])
           f[i] += force
           f[j] -= force

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
   for i in range(n):
       v[i] += dt * f[i] / m[i]
       p[i] += dt * v[i]

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