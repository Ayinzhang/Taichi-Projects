import taichi as ti
ti.init(arch=ti.gpu)

dt=1e-2
max_num=1024
x=ti.Vector.field(2,dtype=ti.f32,shape=max_num)
v=ti.Vector.field(2,dtype=ti.f32,shape=max_num)
f=ti.Vector.field(2,dtype=ti.f32,shape=max_num)
fix=ti.field(dtype=ti.i32,shape=max_num)
y=ti.field(dtype=ti.f32,shape=())
num=ti.field(dtype=ti.i32,shape=())
damp=ti.field(dtype=ti.f32,shape=())
len=ti.field(dtype=ti.f32,shape=(max_num,max_num))
gui=ti.GUI("Mass and sping system",background_color=0xDDDDDD)
y[None]=300
y_label=gui.label("Young's modulus")
damp[None]=1
damp_label=gui.label("Friction coefficien")
friction=ti.field(dtype=ti.f32,shape=())
len=ti.field(dtype=ti.f32,shape=(max_num,max_num))
friction[None]=1

@ti.kernel
def update():
    n=num[None]
    for i in range(n):
        if fix[i]==0:
            f[i]=ti.Vector([0,-9.8])
            for j in range(n):
                if len[i,j]!=0:
                    f[i]-=y[None]*((x[i]-x[j]).norm()/len[i,j]-1)*(x[i]-x[j]).normalized()
    for i in range(n):
        if fix[i]==0:
            v[i]+=dt*f[i]
            v[i]*=ti.exp(-dt*damp[None])
            v[i]*=ti.exp(-dt*damp[None])
            v[i]*=ti.exp(-dt*friction[None])
            x[i]+=dt*v[i]
    for i in range(n):
        for j in range(n):
            if i!=j and (x[i]-x[j]).norm()<0.01 and not fix[i]&fix[j]:
                v[i]+=v[j]
                v[j]=v[i]-v[j]
                v[i]-=v[j]
                v[i]=v[i].norm()*(x[i]-x[j]).normalized()
                v[j]=v[j].norm()*(x[j]-x[i]).normalized()
                if fix[i]:
                    v[i]=ti.Vector([0,0])
                if fix[j]:
                    v[j]=ti.Vector([0,0])

    for i in range(n):
        for j in ti.static(range(2)):
            if x[i][j]<0:
                x[i][j]=0
                v[i][j]=0
            elif x[i][j]>1:
                x[i][j]=1
                v[i][j]=0

@ti.kernel
def add(xpos:ti.f32,ypos:ti.f32,fixed:ti.i32):
    cnt=num[None]
    num[None]+=1
    x[cnt]=ti.Vector([xpos,ypos])
    fix[cnt]=fixed
    for i in range(cnt):
        if (x[i]-x[cnt]).norm()<0.15:
            len[i,cnt]=(x[i]-x[cnt]).norm()
            len[cnt,i]=(x[i]-x[cnt]).norm()

while gui.running:
    update()
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key=='w':
            y[None]*=1.1
        if e.key=='s':
            y[None]/=1.1
        if e.key=='d':
            damp[None]*=1.5
        if e.key=='a':
            damp[None]/=1.5
        if e.key == ti.GUI.LMB:
            add(e.pos[0],e.pos[1],gui.is_pressed(ti.GUI.SHIFT))
        if e.key == ti.GUI.ESCAPE:
            gui.running=False

    y_label.value=y[None]
    damp_label.value=damp[None]

    for i in range (num[None]):
        gui.circle(x[i],color=0x0,radius=5)
    for i in range(num[None]):
        for j in range(num[None]):
            if len[i,j]!=0:
                gui.line(begin=x[i],end=x[j],color=0x888888,radius=min(3,1*len[i,j]/max(0.01,(x[i]-x[j]).norm())))
    gui.show()