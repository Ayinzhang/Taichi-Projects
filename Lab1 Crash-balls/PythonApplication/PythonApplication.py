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
        f[i][0] = wind[None]
        f[i][1] = gravity[None]
        v[i] += dt * f[i]
        p[i] += dt * v[i]
        cross()
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
                iTan = ti.math.normalize(p[j] - p[i]) * ti.math.dot(v[i], ti.math.normalize(p[j] - p[i]))
                jTan = ti.math.normalize(p[i] - p[j]) * ti.math.dot(v[j], ti.math.normalize(p[i] - p[j]))
                v[i] -= 2 * iTan
                v[j] -= 2 * jTan
                p[i] += r * v[i]
                p[j] += r * v[j]
                if ti.math.length(p[i] - p[j]) < 1e-2:
                    l = 0.0
                    r = 1e-2
                    while(r - l > 1e-3):
                        m = (l + r) / 2
                        iPos = m * ti.math.normalize(p[i] - p[j])
                        jPos = m * ti.math.normalize(p[j] - p[i])
                        if ti.math.length(iPos - jPos) < 1e-2:
                            l = m
                        else:
                            r = m
                    p[i] += r * ti.math.normalize(p[i] - p[j])
                    p[j] += r * ti.math.normalize(p[j] - p[i])

@ti.kernel
def implicit_update():
    for i in range(num[None]):
        lp[i] = p[i]
        lv[i] = v[i]
        lf[i] = f[i]
        f[i] = ti.Vector([wind[None], gravity[None]])
        v[i] += dt * f[i]
        p[i] += dt * v[i]
        cross()
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
                iTan = ti.math.normalize(p[j] - p[i]) * ti.math.dot(v[i], ti.math.normalize(p[j] - p[i]))
                jTan = ti.math.normalize(p[i] - p[j]) * ti.math.dot(v[j], ti.math.normalize(p[i] - p[j]))
                v[i] -= 2 * iTan
                v[j] -= 2 * jTan
                p[i] += r * v[i]
                p[j] += r * v[j]
                if ti.math.length(p[i] - p[j]) < 1e-2:
                    l = 0.0
                    r = 1e-2
                    while(r - l > 1e-3):
                        m = (l + r) / 2
                        iPos = m * ti.math.normalize(p[i] - p[j])
                        jPos = m * ti.math.normalize(p[j] - p[i])
                        if ti.math.length(iPos - jPos) < 1e-2:
                            l = m
                        else:
                            r = m
                    p[i] += r * ti.math.normalize(p[i] - p[j])
                    p[j] += r * ti.math.normalize(p[j] - p[i])

@ti.func
def cross():
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

@ti.kernel
def add(xpos:ti.f32, ypos:ti.f32):
    if num[None] < max_num:
        p[num[None]] = ti.Vector([xpos, ypos])
        v[num[None]] = ti.Vector([0, 0])

        num[None] += 1

while gui.running:
    #if num[None] > 0:
    #    print(p[0])
    if update_model[None] == 0:
        explicit_update()
    elif update_model[None] == 1:
        implicit_update()
    elif update_model[None] == 2:
        1
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
            update_model[None] = (update_model[None] + 1) % 3
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
    gui.show()