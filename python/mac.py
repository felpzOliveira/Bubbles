import taichi as ti
import vmath
import grid
import matplotlib.cm as cm

ti.init(arch=ti.gpu)
res = 512
pixels = ti.Vector.field(3, float, shape=(res, res))
#pixels = ti.field(dtype=float, shape=(res,res))
dt = 0.02
pixel_mid = res // 2
ix_length = 15 * res / 512
iy_length = 10
area = ti.Vector([pixel_mid - ix_length, 8, pixel_mid + ix_length, 8 + iy_length])
inflow_velocity = ti.Vector([0.0, 3.0])
inflow_density = ti.Vector([0.7, 0.7, 0.7])

density = 1.0
dx = 1.0
pScale = (dt / dx * density)
pApplyScale = (dt / dx * density)

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
        scale = 1.0#dt / (density * dx)
        pf_ipj = p_at(pf, i, j)
        pf_inj = p_at(pf, i-1, j)
        vf_u_new[i, j] = vf_u[i, j] - (pf_ipj - pf_inj) * scale

@ti.kernel
def apply_pressure_v(vf_v: ti.template(), vf_v_new: ti.template(), pf: ti.template()):
    for i, j, in vf_v:
        scale = 1.0#dt / (density * dx)
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
        divf[i, j] = (div_u + div_v) * scale

@ti.kernel
def velocity_magnitude(vf_u: ti.template(), vf_v: ti.template(), vm: ti.template()):
    for i, j in vm:
        vel_u = (vel_at(vf_u,i+1,j) + vel_at(vf_u,i,j)) * 0.5
        vel_v = (vel_at(vf_v,i,j+1) + vel_at(vf_v,i,j)) * 0.5
        vm[i, j] = ti.sqrt(vel_u ** 2 + vel_v ** 2)

@ti.data_oriented
class Solver:
    def __init__(self, nx, ny):
        self.bSolver = grid.BoundarySolver()
        self.velocity_divs = ti.field(float, shape=(res, res))
        self.diff_pressures = ti.field(float, shape=(res, res))
        self.velocity_slab = grid.Slab(res, res, 2)
        self.pressure_slab = grid.Slab(res, res, 1)
        self.density_slab = grid.Slab(res, res, 3)
        self.velocity_mag = ti.field(float, shape=(res, res))

        self.velocity_u_slab = grid.Slab(res+1, res, 1)
        self.velocity_v_slab = grid.Slab(res, res+1, 1)
        self.velocity_u_slab.set_origin(0.0, 0.5)
        self.velocity_v_slab.set_origin(0.5, 0.0)

        #self.is_mac = 0
        #self.bSolver.setup(self.velocity_slab, self.pressure_slab)

        self.is_mac = 1
        self.bSolver.setup_mac(self.velocity_u_slab, self.velocity_v_slab,
                               self.pressure_slab)

    def handle_inflow(self, area, inflow_vel, inflow_density):
        self.density_slab.set_inflow(area, inflow_density)
        if self.is_mac == 0:
            self.velocity_slab.set_inflow(area, inflow_vel)
        else:
            self.velocity_u_slab.set_inflow(area, inflow_vel[0])
            self.velocity_v_slab.set_inflow(area, inflow_vel[1])

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
    scale = dt / (density * dx * dx)
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
    while residual > 0.0001:
        residual = pressure_jacobi_iter(pf_pair.curr, pf_pair.next, divf)
        pf_pair.flip()
        counter += 1
        if counter > max_it:
            break

@ti.kernel
def fill_color(ipixels: ti.template(), idyef: ti.template()):
    for i, j in ipixels:
        density = ti.min(1.0, ti.max(0.0, idyef[i, j]))
        ipixels[i, j] = density

def fill_color_vel():
    #fill_color_vel_kern(ipixels, vf)
    velocity_magnitude(solver.velocity_u_slab.curr, solver.velocity_v_slab.curr,
                       solver.velocity_mag)
    V_np = solver.velocity_mag.to_numpy()
    return cm.jet(V_np)

gui = ti.GUI('Advection schemes', (res, res))
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: exit(0)
    for itr in range(15):
        # Add inflow:
        solver.handle_inflow(area, inflow_velocity, inflow_density)
        # Advection:
        solver.velocity_bc()

        solver.advect_velocities(dt)
        solver.advect_density(dt)
        solver.flip_velocities()
        solver.density_slab.flip()

        solver.velocity_bc()
        # External forces:

        # Projection:
        solver.divergence()
        pressure_jacobi(solver.pressure_slab, solver.velocity_divs)

        solver.pressure_bc()

        solver.apply_pressure()
        solver.flip_velocities()

    # Put color from dye to pixel:
    fill_color(pixels, solver.density_slab.curr)
    gui.set_image(pixels)
    #V_img = fill_color_vel()
    #gui.set_image(V_img)
    gui.show()

