import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)

res = 512
paused = False
dt = 0.03
p_jacobi_iters = 500
f_strength = 10000.0
curl_strength = 0
dye_decay = 1 - 1 / 60
force_radius = res / 2.0

class Pair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

_velocities = ti.Vector.field(2, float, shape=(res, res))
_new_velocities = ti.Vector.field(2, float, shape=(res, res))
velocity_divs = ti.field(float, shape=(res, res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
velocities_pair = Pair(_velocities, _new_velocities)
pressures_pair = Pair(_pressures, _new_pressures)
dyes_pair = Pair(_dye_buffer, _new_dye_buffer)

class GenMouseData:
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = (mxy - self.prev_mouse) 
                mdir /= np.linalg.norm(mdir) + 1e-5
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data

@ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)

@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]

@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def backtrace(vf, p, dt_):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp(vf, p2)
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p

@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay

@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(), imp_data: ti.types.ndarray()):
    g_dir = -ti.Vector([0, 9.8]) * 300
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        factor = ti.exp(-d2 / force_radius)

        a = dyef[i, j].norm()
        vf[i, j] += (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt

        if mdir.norm() > 0.5:
            dyef[i, j] += ti.exp(-d2 * (4 / (res / 15) ** 2)) * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])

@ti.kernel
def apply_velocity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5

@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb), abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = ti.min(ti.max(vf[i, j] + force * dt, -1e3), 1e3)

@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25

def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])

def update(mouse_data):
    #Step 1:Advection
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()
    #Step 2:Applying forces
    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)
    apply_velocity(velocities_pair.cur)
    enhance_vorticity(velocities_pair.cur, velocity_curls)
    #Step 3:Projection
    solve_pressure_jacobi()
    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

gui = ti.GUI("Stable Fluid", (res, res))
md_gen = GenMouseData()

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.ESCAPE:
            break
        elif e.key == "r":
            dyes_pair.cur.fill(0)
        elif e.key == "p":
            paused = not paused

    if not paused:
        mouse_data = md_gen(gui)
        update(mouse_data)
        gui.set_image(dyes_pair.cur)
    gui.show()