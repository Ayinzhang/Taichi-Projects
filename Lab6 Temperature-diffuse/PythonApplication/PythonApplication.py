import taichi as ti
ti.init(ti.cpu)

k = 3000
n = 512
dt = 1e-2
radius = 64
t_min = 0
t_max = 300
paused = False
n2 = n * n
c = dt * k
t = ti.field(ti.f32, n2)
pixels = ti.Vector.field(3, ti.f32, (n, n))
D_Builder = ti.linalg.SparseMatrixBuilder(n2, n2, 5 * n2)

@ti.kernel
def fillDiffusionMatrixBuilder(A: ti.types.sparse_matrix_builder()):
    for i,j in ti.ndrange(n, n):
        num = 0.0
        if i - 1 >= 0:
            A[i * n + j, (i - 1) * n + j] -= c
            num += c
        if i + 1 < n:
            A[i * n + j, (i + 1) * n + j] -= c
            num += c
        if j - 1 >= 0:
            A[i * n + j, i * n + j - 1] -= c
            num += c
        if j + 1 < n:
            A[i * n + j, i * n + j + 1] -= c
            num += c
        A[i * n + j, i * n + j] += num + 1

@ti.kernel
def init():
    for i,j in ti.ndrange(n, n):
        if (i - n // 2) ** 2 + (j - n // 2) ** 2 <= radius ** 2:
            t[i * n + j] = t_max
        else:
            t[i * n + j] = t_min

@ti.kernel
def update_sourse():
    for i in range(n // 2 - radius, n // 2 + radius + 1):
        for j in range(n // 2 - radius, n // 2 + radius + 1):
            if (i - n // 2) ** 2 + (j - n // 2) ** 2 <= radius ** 2:
                t[i * n + j] = t_max

@ti.kernel
def cook(t: ti.template(), color: ti.template(), t_min: ti.f32, t_max: ti.f32):
    for i,j in ti.ndrange(n, n):
        color[i, j] = ti.Vector([1, 1, 1])
        d = t_max - t_min
        if t[i * n + j] < (t_min + 0.25 * d):
            color[i, j][0] = 0
            color[i, j][1] = 4 * (t[i * n + j] - t_min) / d
        elif t[i * n + j] < (t_min + 0.5 * d):
            color[i, j][0] = 0
            color[i, j][2] = 1 + 4 * (t_min + 0.25 * d - t[i * n + j]) / d
        elif t[i * n + j] < (t_min + 0.75 * d):
            color[i, j][0] = 4 * (t[i * n + j] - t_min - 0.5 * d) / d
            color[i, j][2] = 0
        else:
            color[i, j][1] = 1 + 4 * (t_min + 0.75 * d - t[i * n + j]) / d
            color[i, j][2] = 0

gui = ti.GUI("Temperature-diffuse", (n, n))
init()
fillDiffusionMatrixBuilder(D_Builder)
D = D_Builder.build()
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(D)
solver.factorize(D)
ti.profiler.clear_kernel_profiler_info()
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == 'r':
            init()
    if not paused:
        t.from_numpy(solver.solve(t))
        update_sourse()
        cook(t, pixels, t_min, t_max)
    gui.set_image(pixels)
    gui.show()