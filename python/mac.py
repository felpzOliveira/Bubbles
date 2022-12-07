import taichi as ti
import core.vmath
import core.grid
import matplotlib.cm as cm

ti.init(arch=ti.gpu)
res = 512
pixels = ti.Vector.field(3, float, shape=(res, res))
#pixels = ti.field(dtype=float, shape=(res,res))
timestep = 0.005
pixel_mid = res // 2
ix_length = 15 * res / 512
iy_length = 10
area = ti.Vector([pixel_mid - ix_length, 8, pixel_mid + ix_length, 8 + iy_length])
#area = ti.Vector([8, pixel_mid - ix_length, 8 + iy_length, pixel_mid + ix_length])
inflow_velocity = ti.Vector([0.0, 0.0])
inflow_density = ti.Vector([0.7, 0.7, 0.7])

density = 1.0
dx = 1.0

@ti.kernel
def apply_pressure_u(vf_u: ti.template(), vf_u_new: ti.template(), pf: ti.template()):
    for i, j in vf_u:
        scale = timestep / (density * dx)
        pf_ipj = core.grid.qf_near_value(pf, i, j)
        pf_inj = core.grid.qf_near_value(pf, i-1, j)
        vf_u_new[i, j] = vf_u[i, j] - (pf_ipj - pf_inj) * scale

@ti.kernel
def apply_pressure_v(vf_v: ti.template(), vf_v_new: ti.template(), pf: ti.template()):
    for i, j, in vf_v:
        scale = timestep / (density * dx)
        pf_ipj = core.grid.qf_near_value(pf, i, j)
        pf_inj = core.grid.qf_near_value(pf, i, j-1)
        vf_v_new[i, j] = vf_v[i, j] - (pf_ipj - pf_inj) * scale

@ti.kernel
def apply_temperature(vf_v: ti.template(), den: ti.template(),
                          tp: ti.template(), dt: float, tamb: float):
    for i, j in vf_v:
        p = ti.Vector([i, j]) + ti.Vector([0.5, 0.0])
        d = core.grid.qf_sample_value(den, p, 0.5, 0.5)[0]
        tempij = core.grid.qf_sample_value(tp, p, 0.5, 0.5)
        alpha = 0.0006
        beta = 5.0
        up = 1
        if tempij > tamb:
            delta_temp = tempij - tamb
            fbuo = -alpha * d + beta * delta_temp
            vf_v[i, j] += fbuo * dt * up

@ti.kernel
def decay_temperature(tp: ti.template(), alpha: float):
    for i, j in tp:
        tp[i, j] *= ti.max(0.0, (1.0 - alpha))

@ti.data_oriented
class Solver:
    def __init__(self, nx, ny):
        self.bSolver = core.grid.BoundarySolver()
        self.pressure_slab = core.grid.Slab(res, res, 1)
        self.density_slab = core.grid.Slab(res, res, 3)
        self.temperature_slab = core.grid.Slab(res, res, 1)

        self.velocity_u_slab = core.grid.Slab(res+1, res, 1)
        self.velocity_v_slab = core.grid.Slab(res, res+1, 1)
        self.velocity_u_slab.set_origin(0.0, 0.5)
        self.velocity_v_slab.set_origin(0.5, 0.0)

        self.bSolver.setup(self.velocity_u_slab, self.velocity_v_slab)

    def handle_inflow(self, area, inflow_vel, inflow_density):
        self.density_slab.set_inflow(area, inflow_density)
        self.velocity_u_slab.set_inflow(area, inflow_vel[0])
        self.velocity_v_slab.set_inflow(area, inflow_vel[1])
        self.temperature_slab.set_inflow(area, 600)

    def advect_velocities(self, dt):
        self.velocity_u_slab.advect(self.velocity_u_slab.curr,
                                    self.velocity_v_slab.curr, dt)
        self.velocity_v_slab.advect(self.velocity_u_slab.curr,
                                    self.velocity_v_slab.curr, dt)

    def advect_density(self, dt):
        self.density_slab.advect(self.velocity_u_slab.curr, self.velocity_v_slab.curr, dt)

    def advect_temperature(self, dt):
        self.temperature_slab.advect(self.velocity_u_slab.curr,
                                     self.velocity_v_slab.curr, dt)

    def velocity_bc(self):
        self.bSolver.update_velocity()

    def flip_velocities(self):
        self.velocity_u_slab.flip()
        self.velocity_v_slab.flip()

    def apply_temperature(self, dt):
        decay_temperature(self.temperature_slab.curr, 0.02)
        tamb = self.temperature_slab.average()
        apply_temperature(self.velocity_v_slab.curr, self.density_slab.curr,
                          self.temperature_slab.curr, dt, tamb)

    def apply_pressure(self):
        apply_pressure_u(self.velocity_u_slab.curr, self.velocity_u_slab.next,
                         self.pressure_slab.curr)
        apply_pressure_v(self.velocity_v_slab.curr, self.velocity_v_slab.next,
                         self.pressure_slab.curr)

solver = Solver(res, res)

@ti.kernel
def fill_color(ipixels: ti.template(), idyef: ti.template()):
    for i, j in ipixels:
        density = ti.min(1.0, ti.max(0.0, idyef[i, j]))
        ipixels[i, j] = density

gui = ti.GUI('Basic Smoke Solver 2D', (res, res))
#pSolver = core.grid.RedBlackSORSolver(solver.pressure_slab)
pSolver = core.grid.JacobiSolver(solver.pressure_slab)

while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: exit(0)
    for itr in range(15):
        # Add inflow:
        solver.handle_inflow(area, inflow_velocity, inflow_density)
        # Advection:
        solver.velocity_bc()

        solver.advect_velocities(timestep)
        solver.advect_density(timestep)
        solver.advect_temperature(timestep)
        solver.flip_velocities()
        solver.temperature_slab.flip()
        solver.density_slab.flip()

        # External forces:
        solver.apply_temperature(timestep)
        solver.velocity_bc()
        # Projection:
        pSolver.update(solver.velocity_u_slab, solver.velocity_v_slab, timestep)

        solver.apply_pressure()
        solver.flip_velocities()

    # Put color from dye to pixel:
    fill_color(pixels, solver.density_slab.curr)
    gui.set_image(pixels)
    gui.show()

