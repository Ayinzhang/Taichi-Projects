import taichi as ti
ti.init(arch=ti.gpu)

dt = 1e-2
max_num = 64
num = ti.field(dtype = ti.i32, shape=())
wind = ti.field(dtype = ti.f32, shape=())
gravity = ti.field(dtype = ti.f32, shape=())
damp = ti.field(dtype = ti.f32, shape=())
update_model = ti.field(dtype = ti.i32, shape=())
p = ti.Vector.field(2, dtype = ti.f32, shape = max_num)
v = ti.Vector.field(2, dtype = ti.f32, shape = max_num)
f = ti.Vector.field(2, dtype = ti.f32, shape = max_num)
lp = ti.Vector.field(2, dtype = ti.f32, shape = max_num)
lv = ti.Vector.field(2, dtype = ti.f32, shape = max_num)
lf = ti.Vector.field(2, dtype = ti.f32, shape = max_num)
gui = ti.GUI("Crash Balls", (1080, 640), background_color=0xDDDDDD)
gravity[None] = -9.8
damp[None] = 0.15
wind_label = gui.label("Wind")
gravity_label = gui.label("Gravity")
damp_label = gui.label("Damp")

@ti.kernel
def explicit_update():
    for i in range(num[None]):
        lp[i] = p[i]
        lv[i] = v[i]
        lf[i] = f[i]
        f[i] = ti.Vector([wind[None], gravity[None]])
        v[i] += dt * lf[i]
        p[i] += dt * lv[i]

@ti.kernel
def implicit_update():
    for i in range(num[None]):
        lp[i] = p[i]
        lv[i] = v[i]
        lf[i] = f[i]
        f[i] = ti.Vector([wind[None], gravity[None]])
        v[i] += dt * f[i]
        p[i] += dt * v[i]

@ti.kernel
def semi_implicit_update():
    for i in range(num[None]):
        lp[i] = p[i]
        lv[i] = v[i]
        lf[i] = f[i]
        f[i] = ti.Vector([wind[None], gravity[None]])
        v[i] += dt * lf[i]
        p[i] += dt * v[i]

@ti.kernel
def velocity_welley_update():
    for i in range(num[None]):
        lp[i] = p[i]
        lv[i] = v[i]
        lf[i] = f[i]
        f[i] = ti.Vector([wind[None], gravity[None]])
        p[i] += dt * v[i] + f[i] * dt ** 2 / 2
        v[i] += dt * (lf[i] + f[i]) / 2

@ti.kernel
def runge_kutta2():
    for i in range(num[None]):
        lp[i] = p[i]
        lv[i] = v[i]
        lf[i] = f[i]
        f[i] = ti.Vector([wind[None], gravity[None]])
        v[i] += dt * lf[i]
        p[i] += dt * lv[i] / 2
        f[i] = ti.Vector([wind[None], gravity[None]])
        v[i] += dt * lf[i]
        p[i] += dt * lv[i]

@ti.kernel
def update():
    for i in range(num[None]):
        for j in ti.static(range(2)):
            if p[i][j] < 0:
                if ti.math.ceil(-p[i][j]) % 2 == 1:
                    p[i][j] = ti.math.fract(-p[i][j])
                    v[i][j] = - (1 - damp[None]) * v[i][j]
                else:
                    p[i][j] = ti.math.fract(p[i][j])
                    v[i][j] = (1 - damp[None]) * v[i][j]
            elif p[i][j] > 1:
                if ti.math.floor(p[i][j]) % 2 == 1:
                    p[i][j] = ti.math.fract(-p[i][j])
                    v[i][j] = - (1 - damp[None]) * v[i][j]
                else:
                    p[i][j] = ti.math.fract(p[i][j])
                    v[i][j] = (1 - damp[None]) * v[i][j]
    for i in range(num[None]):
        for j in range(i + 1, num[None]):
            if ti.math.length(p[i] - p[j])< 1e-2:
                l = 0.0
                r = dt
                while(r - l > 1e-5):
                    m = (l + r) / 2
                    iPos = p[i] - m * v[i]
                    jPos = p[j] - m * v[j]
                    if ti.math.length(iPos - jPos) < 1e-2:
                        l = m
                    else:
                        r = m
                p[i] -= r * v[i]
                p[j] -= r * v[j]
                iTan = (p[j] - p[i]).normalized() * (p[j] - p[i]).normalized().dot(v[i])
                jTan = (p[i] - p[j]).normalized() * (p[i] - p[j]).normalized().dot(v[j])
                v[i] = v[i] - iTan
                v[j] = v[j] - jTan
                p[i] += r * v[i]
                p[j] += r * v[j]


@ti.kernel
def add(xpos:ti.f32, ypos:ti.f32):
    if num[None] < max_num:
        p[num[None]] = ti.Vector([xpos, ypos])
        v[num[None]] = ti.Vector([0, 0])
        f[num[None]] = ti.Vector([0, 0])
        lp[num[None]] = p[num[None]]
        lv[num[None]] = v[num[None]]
        lf[num[None]] = f[num[None]]
        num[None] += 1

while gui.running:
    if update_model[None] == 0:
        explicit_update()
    elif update_model[None] == 1:
        implicit_update()
    elif update_model[None] == 2:
        semi_implicit_update()
    elif update_model[None] == 3:
        velocity_welley_update()
    elif update_model[None] == 4:
         runge_kutta2()
    update()
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == 'a' or e.key == ti.GUI.LEFT:
            wind[None] -= 0.5
        if e.key == 'd' or e.key == ti.GUI.RIGHT:
            wind[None] += 0.5
        if e.key == 'w':
            gravity[None] += 0.5
        if e.key == 's':
            gravity[None] -= 0.5
        if e.key == 'r':
            num[None] = 0
        if e.key == 'q':
            update_model[None] = (update_model[None] + 1) % 5
        if e.key == ti.GUI.UP:
            damp[None] += 0.05
        if e.key == ti.GUI.DOWN:
            damp[None] -= 0.05
        if e.key == ti.GUI.LMB:
            add(e.pos[0], e.pos[1])
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
    for i in range(num[None]):
        gui.circle(p[i],color=0x0,radius=5)
    wind_label.value = wind[None]
    gravity_label.value = gravity[None]
    damp_label.value = damp[None]
    if update_model[None] == 0:
        gui.text("Explicit Euler",(0,1), 16, 0x00)
    elif update_model[None] == 1:
        gui.text("Implicit Euler",(0,1), 16, 0x00)
    elif update_model[None] == 2:
        gui.text("Semi-Implicit Euler",(0,1), 16, 0x00)
    elif update_model[None] == 3:
        gui.text("Velocity Welley",(0,1), 16, 0x00)
    elif update_model[None] == 4:
        gui.text("Runge Kutta2",(0,1), 16, 0x00)
    gui.show()