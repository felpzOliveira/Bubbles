import taichi as ti
import core.vmath
import core.grid

@ti.kernel
def apply_pressure_vel2D(vf: ti.template(), vf_new: ti.template(), pf: ti.template(),
                         ox: int, oy: int, timestep: float, density: float, dx: float):
    for i, j in vf:
        scale = timestep / (1.25 * 0.1)
        pf_ipj = core.grid.qf_near_value2D(pf, i, j)
        pf_inj = core.grid.qf_near_value2D(pf, i - ox, j - oy)
        vf_new[i, j] = vf[i, j] - (pf_ipj - pf_inj) * scale

@ti.kernel
def apply_temperature2D(vf_v: ti.template(), den: ti.template(), tp: ti.template(),
                        dt: float, tamb: float, alpha: float, beta: float):
    for i, j in vf_v:
        p = ti.Vector([i, j]) + ti.Vector([0.5, 0.0])
        d = core.grid.qf_sample_value2D(den, p, 0.5, 0.5)
        tempij = core.grid.qf_sample_value2D(tp, p, 0.5, 0.5)
        up = 1
        if tempij > tamb:
            delta_temp = tempij - tamb
            fbuo = -alpha * d + beta * delta_temp
            vf_v[i, j] += fbuo * dt * up

@ti.kernel
def decay_temperature2D(tp: ti.template(), alpha: float):
    for i, j in tp:
        tp[i, j] *= ti.max(0.0, (1.0 - alpha))

@ti.kernel
def fill_pixels2D(pixel_buffer: ti.template(), density: ti.template(),
                  bSolver: ti.template()):
    for i, j in pixel_buffer:
        if bSolver.is_wall(i, j):
            pixel_buffer[i, j] = ti.Vector([0.5, 0.7, 0.5])
        else:
            val = ti.min(1.0, ti.max(0.0, density[i, j]))
            pixel_buffer[i, j] = ti.Vector([val, val, val])


@ti.data_oriented
class SmokeSolver2D:
    def __init__(self, nx, ny, params):
        self.bSolver = core.grid.BoundarySolver2D()
        self.pressure_slab = core.grid.Slab2D(nx, ny, 1)
        self.density_slab = core.grid.Slab2D(nx, ny, 1)
        self.temperature_slab = core.grid.Slab2D(nx, ny, 1)

        self.velocity_u_slab = core.grid.Slab2D(nx+1, ny, 1)
        self.velocity_v_slab = core.grid.Slab2D(nx, ny+1, 1)
        self.velocity_u_slab.set_origin(0.0, 0.5)
        self.velocity_v_slab.set_origin(0.5, 0.0)

        self.bSolver.setup(self.velocity_u_slab, self.velocity_v_slab)
        self.img_pixels = ti.Vector.field(3, float, shape=(nx, ny))

        self.params = params

    def handle_inflow(self, area, inflow_vel, inflow_density):
        self.density_slab.set_inflow(area, inflow_density)
        self.velocity_u_slab.set_inflow(area, inflow_vel[0])
        self.velocity_v_slab.set_inflow(area, inflow_vel[1])
        self.temperature_slab.set_inflow(area, 600)

    def set_pressure_solver(self, pSolver):
        self.pSolver = pSolver
        self.pSolver.set_boundary_solver(self.bSolver)

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
        decay_temperature2D(self.temperature_slab.curr, self.params.temperature_decay())
        tamb = self.temperature_slab.average()
        apply_temperature2D(self.velocity_v_slab.curr, self.density_slab.curr,
                            self.temperature_slab.curr, dt, tamb, self.params.alpha(),
                            self.params.beta())

    def solve_pressure(self, dt):
        self.pSolver.update(self.velocity_u_slab, self.velocity_v_slab, dt)

    def apply_pressure(self, timestep, density, dx):
        apply_pressure_vel2D(self.velocity_u_slab.curr, self.velocity_u_slab.next,
                             self.pressure_slab.curr, 1, 0, timestep, density, dx)
        apply_pressure_vel2D(self.velocity_v_slab.curr, self.velocity_v_slab.next,
                             self.pressure_slab.curr, 0, 1, timestep, density, dx)

    def flat_image(self):
        fill_pixels2D(self.img_pixels, self.density_slab.curr, self.bSolver)

    def pixels(self):
        return self.img_pixels

    def dimensions(self):
        return 2

