import taichi as ti
import vmath
import grid
import matplotlib.cm as cm

ti.init(arch=ti.gpu)
res = 512
pixels = ti.Vector.field(3, float, shape=(res, res))
#pixels = ti.field(dtype=float, shape=(res,res))
timestep = 0.01
pixel_mid = res // 2
ix_length = 15 * res / 512
iy_length = 10
area = ti.Vector([pixel_mid - ix_length, 8, pixel_mid + ix_length, 8 + iy_length])
inflow_velocity = ti.Vector([0.0, 3.0])
inflow_density = ti.Vector([0.7, 0.7, 0.7])

density = 1.0
dx = 1.0
USE_MAC_GRID = 1

@ti.kernel
def apply_pressure(vf: ti.template(), vf_new: ti.template(), pf: ti.template()):
    for i, j in vf:
        pf_ipj = p_at(pf, i+1, j)
        pf_inj = p_at(pf, i-1, j)
        pf_ijp = p_at(pf, i, j+1)
        pf_ijn = p_at(pf, i, j-1)
        vf_new[i, j] = vf[i, j] - ti.Vector([pf_ipj - pf_inj, pf_ijp - pf_ijn])

@ti.kernel
def apply_pressure_u(vf_u: ti.template(), vf_u_new: ti.template(), pf: ti.template()):
    for i, j in vf_u:
        scale = timestep / (density * dx)
        pf_ipj = p_at(pf, i, j)
        pf_inj = p_at(pf, i-1, j)
        vf_u_new[i, j] = vf_u[i, j] - (pf_ipj - pf_inj) * scale

@ti.kernel
def apply_pressure_v(vf_v: ti.template(), vf_v_new: ti.template(), pf: ti.template()):
    for i, j, in vf_v:
        scale = timestep / (density * dx)
        pf_ipj = p_at(pf, i, j)
        pf_inj = p_at(pf, i, j-1)
        vf_v_new[i, j] = vf_v[i, j] - (pf_ipj - pf_inj) * scale

@ti.kernel
def divergence(vf: ti.template(), divf: ti.template()):
    for i, j in divf:
        scale = 1.0#dt / (density * dx)
        div_u = vel_at(vf,i+1,j)[0] - vel_at(vf,i-1,j)[0]
        div_v = vel_at(vf,i,j+1)[1] - vel_at(vf,i,j-1)[1]
        divf[i, j] = 0.5 * (div_u + div_v) * scale


@ti.kernel
def divergence_mac(vf_u: ti.template(), vf_v: ti.template(), divf: ti.template()):
    for i, j in divf:
        scale = 1.0#1.0 / dx
        div_u = vel_at(vf_u,i+1,j) - vel_at(vf_u,i,j)
        div_v = vel_at(vf_v,i,j+1) - vel_at(vf_v,i,j)
        divf[i, j] = 0.5 * (div_u + div_v) * scale

@ti.kernel
def velocity_magnitude(vf_u: ti.template(), vf_v: ti.template(), vm: ti.template()):
    for i, j in vm:
        vel_u = (vel_at(vf_u,i+1,j) + vel_at(vf_u,i,j)) * 0.5
        vel_v = (vel_at(vf_v,i,j+1) + vel_at(vf_v,i,j)) * 0.5
        vm[i, j] = ti.sqrt(vel_u ** 2 + vel_v ** 2)

@ti.kernel
def apply_temperature(vf: ti.template(), den: ti.template(),
                      tp: ti.template(), dt: float, tamb: float):
    for i, j in vf:
        p = ti.Vector([i, j]) + ti.Vector([0.5, 0.5])
        d = grid.qf_sample_value(den, p, 0.5, 0.5)[0]
        tempij = grid.qf_sample_value(tp, p, 0.5, 0.5)
        alpha = 0.0006
        beta = 5.0
        up = ti.Vector([0.0, 1.0])
        if tempij > tamb:
            delta_temp = tempij - tamb
            fbuo = -alpha * d + beta * delta_temp
            vf[i, j] += fbuo * dt * up

@ti.kernel
def apply_temperature_mac(vf_v: ti.template(), den: ti.template(),
                          tp: ti.template(), dt: float, tamb: float):
    for i, j in vf_v:
        p = ti.Vector([i, j]) + ti.Vector([0.5, 0.0])
        d = grid.qf_sample_value(den, p, 0.5, 0.5)[0]
        tempij = grid.qf_sample_value(tp, p, 0.5, 0.5)
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
        self.bSolver = grid.BoundarySolver()
        self.velocity_divs = ti.field(float, shape=(res, res))
        self.velocity_slab = grid.Slab(res, res, 2)
        self.pressure_slab = grid.Slab(res, res, 1)
        self.density_slab = grid.Slab(res, res, 3)
        self.velocity_mag = ti.field(float, shape=(res, res))
        self.temperature_slab = grid.Slab(res, res, 1)

        self.velocity_u_slab = grid.Slab(res+1, res, 1)
        self.velocity_v_slab = grid.Slab(res, res+1, 1)
        self.velocity_u_slab.set_origin(0.0, 0.5)
        self.velocity_v_slab.set_origin(0.5, 0.0)

        if USE_MAC_GRID == 0:
            self.bSolver.setup(self.velocity_slab, self.pressure_slab)
        else:
            self.bSolver.setup_mac(self.velocity_u_slab, self.velocity_v_slab,
                                   self.pressure_slab)

        self.is_mac = USE_MAC_GRID

    def handle_inflow(self, area, inflow_vel, inflow_density):
        self.density_slab.set_inflow(area, inflow_density)
        if self.is_mac == 0:
            self.velocity_slab.set_inflow(area, inflow_vel)
        else:
            self.velocity_u_slab.set_inflow(area, inflow_vel[0])
            self.velocity_v_slab.set_inflow(area, inflow_vel[1])

        self.temperature_slab.set_inflow(area, 100)

    def advect_velocities(self, dt):
        if self.is_mac == 0:
            self.velocity_slab.advect(self.velocity_slab.curr, dt)
        else:
            self.velocity_u_slab.advect_mac(self.velocity_u_slab.curr,
                                            self.velocity_v_slab.curr, dt)
            self.velocity_v_slab.advect_mac(self.velocity_u_slab.curr,
                                            self.velocity_v_slab.curr, dt)

    def advect_density(self, dt):
        if self.is_mac == 0:
            self.density_slab.advect(self.velocity_slab.curr, dt)
        else:
            self.density_slab.advect_mac(self.velocity_u_slab.curr,
                                         self.velocity_v_slab.curr, dt)

    def advect_temperature(self, dt):
        if self.is_mac == 0:
            self.temperature_slab.advect(self.velocity_slab.curr, dt)
        else:
            self.temperature_slab.advect_mac(self.velocity_u_slab.curr,
                                             self.velocity_v_slab.curr, dt)

    def velocity_bc(self):
        self.bSolver.update_velocity()

    def pressure_bc(self):
        self.bSolver.update_pressure()

    def flip_velocities(self):
        if self.is_mac == 0:
            self.velocity_slab.flip()
        else:
            self.velocity_u_slab.flip()
            self.velocity_v_slab.flip()

    def apply_temperature(self, dt):
        decay_temperature(self.temperature_slab.curr, 0.03)
        tamb = self.temperature_slab.average()
        if self.is_mac == 0:
            apply_temperature(self.velocity_slab.curr, self.density_slab.curr,
                              self.temperature_slab.curr, dt, tamb)
        else:
            apply_temperature_mac(self.velocity_v_slab.curr, self.density_slab.curr,
                                  self.temperature_slab.curr, dt, tamb)

    def apply_pressure(self):
        if self.is_mac == 0:
            apply_pressure(self.velocity_slab.curr, self.velocity_slab.next,
                           self.pressure_slab.curr)
        else:
            apply_pressure_u(self.velocity_u_slab.curr, self.velocity_u_slab.next,
                             self.pressure_slab.curr)
            apply_pressure_v(self.velocity_v_slab.curr, self.velocity_v_slab.next,
                             self.pressure_slab.curr)

    def divergence(self):
        if self.is_mac == 0:
            divergence(self.velocity_slab.curr, self.velocity_divs)
        else:
            divergence_mac(self.velocity_u_slab.curr, self.velocity_v_slab.curr,
                           self.velocity_divs)

solver = Solver(res, res)

@ti.func
def vel_at(vf: ti.template(), i: int, j: int):
    ret = 0.0
    #ret = ti.Vector([0.0, 0.0])
    nx = vf.shape[0]
    ny = vf.shape[1]
    if not ((i < 0) or (j < 0) or (i >= nx) or (j >= ny)):
        ret = vf[i, j]
    return ret

@ti.func
def p_at(pf: ti.template(), i: int, j: int):
    ret = 0.0
    if not ((i < 0) or (j < 0) or (i >= res) or (j >= res)):
        ret = pf[i, j]
    return pf[i, j]

@ti.func
def p_bounds(pf: ti.template(), i: int, j: int) -> ti.f32:
    ret = 0.0
    if (i == j == 0) or (i == j == res - 1) or (i == 0 and j == res - 1) or (
            i == res - 1 and j == 0):
        pf[i, j] = 0.0
    elif i == 0:
        pf[i, j] = pf[i+1, j]
    elif j == 0:
        pf[i, j] = pf[i, j+1]
    elif i == res - 1:
        pf[i, j] = pf[i-1, j]
    elif j == res - 1:
        pf[i, j] = pf[i, j-1]

    if (i < 0) or (j < 0) or (i >= res) or (j >= res):
        ret = 0.0
    else:
        ret = pf[i, j]
    return ret

@ti.kernel
def pressure_jacobi_iter(pf: ti.template(), pf_new: ti.template(), divf: ti.template()) -> ti.f32:
    norm_new = 0.0
    norm_diff = 0.0
    for i, j in pf:
        pf_new[i, j] = 0.25 * (p_bounds(pf, i+1, j) + p_bounds(pf, i-1, j) +
                               p_bounds(pf, i, j+1) + p_bounds(pf, i, j-1) - divf[i, j])
        pf_diff = ti.abs(pf_new[i, j] - p_bounds(pf, i, j))
        norm_new += (pf_new[i, j] * pf_new[i, j])
        norm_diff += (pf_diff * pf_diff)
    residual = ti.sqrt(norm_diff / norm_new)
    if norm_new == 0:
        residual = 0.0
    return residual

def pressure_jacobi(pf_pair, divf: ti.template()):
    residual = 10.0
    counter = 0
    max_it = 1
    #pf_pair.curr.fill(0.0)
    while residual > 0.0001:
        residual = pressure_jacobi_iter(pf_pair.curr, pf_pair.next, divf)
        pf_pair.flip()
        counter += 1
        if counter > max_it:
            break
    print("Res = " + str(residual))

@ti.kernel
def fill_color(ipixels: ti.template(), idyef: ti.template()):
    for i, j in ipixels:
        density = ti.min(1.0, ti.max(0.0, idyef[i, j]))
        ipixels[i, j] = density

def fill_color_vel():
    #fill_color_vel_kern(ipixels, vf)
    #velocity_magnitude(solver.velocity_u_slab.curr, solver.velocity_v_slab.curr,
                       #solver.velocity_mag)
    #V_np = solver.velocity_mag.to_numpy()
    V_np = solver.temperature_slab.curr.to_numpy()
    return cm.jet(V_np)

gui = ti.GUI('Basic Smoke Solver 2D', (res, res))
pSolver = grid.RedBlackSORSolver(solver.pressure_slab)
#pSolver = grid.JacobiSolver(solver.pressure_slab)

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
        #solver.divergence()
        #pressure_jacobi(solver.pressure_slab, solver.velocity_divs)

        #solver.pressure_bc()
        pSolver.update(solver.velocity_u_slab, solver.velocity_v_slab, timestep)

        solver.apply_pressure()
        solver.flip_velocities()

    # Put color from dye to pixel:
    fill_color(pixels, solver.density_slab.curr)
    gui.set_image(pixels)
    #V_img = fill_color_vel()
    #gui.set_image(V_img)
    gui.show()

