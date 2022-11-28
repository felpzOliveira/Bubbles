import taichi as ti
import vmath

################################################################################
# Buffer manipulation
################################################################################

# return the value of quantity field 'qf' at location u, v, clamped if required
@ti.func
def qf_near_value(qf: ti.template(), u, v):
    i = max(0, min(int(u), qf.shape[0]-1))
    j = max(0, min(int(v), qf.shape[1]-1))
    return qf[i, j]

# samples a value of quantity field 'qf' at coordinate p = (u,v)
# note that this is done in grid coordinates where grid spacing is dx = 1
# ox, oy account for where is the measure made in 'qf', i.e.:
# for regular grids values are measured in the center with ox = oy = 0.5
# for mac grids these differ for the u component ox = 0.0, oy = 0.5
# and v component ox = 0.5, oy = 0.0
@ti.func
def qf_sample_value(qf: ti.template(), p: ti.template(), ox: float, oy: float):
    s, t = p[0] - ox, p[1] - oy
    iu, iv = int(s), int(t)
    fu, fv = s - iu, t - iv
    f00 = qf_near_value(qf, iu + 0, iv + 0)
    f10 = qf_near_value(qf, iu + 1, iv + 0)
    f01 = qf_near_value(qf, iu + 0, iv + 1)
    f11 = qf_near_value(qf, iu + 1, iv + 1)
    return vmath.bilerp(f00, f10, f01, f11, fu, fv)

# do advection over the velocity field 'vf' of the quantity 'qf' under timestep
# '_dt' and store the results under 'new_qf'. Advection is performed using RK3
# backtrace. This assumes that both the vector field vf (2d) and the quantity qf
# are measured at the center with ox = oy = 0.5
@ti.kernel
def advect_rk3(vf: ti.template(), qf: ti.template(),
               new_qf: ti.template(), _dt: float):
    for i, j in qf:
        p = ti.Vector([i, j]) + ti.Vector([0.5, 0.5])
        k1 = qf_sample_value(vf, p, 0.5, 0.5)
        k2 = qf_sample_value(vf, p - 0.50 * _dt * k1, 0.5, 0.5)
        k3 = qf_sample_value(vf, p - 0.75 * _dt * k2, 0.5, 0.5)
        p_prev = p - (_dt / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3)
        new_qf[i, j] = qf_sample_value(qf, p_prev, 0.5, 0.5)

# do advection over the velocity field described by explicit components 'u' and 'v'
# considering the offset of data points in the layout of the MAC grid. Quantity 'qf'
# is allowed to have any offset 'ox', 'oy' in the range [0.0, 1.0]. Note that this
# routine (like others) work on grid coordinates and not world coordinates, it is
# assumed that grids are on top of each other and no translation/scaling is present
@ti.kernel
def advect_rk3_mac(vf_u: ti.template(), vf_v: ti.template(),
                   qf: ti.template(), new_qf: ti.template(), _dt: float,
                   ox: float, oy: float):
    for i,j in qf:
        p = ti.Vector([i, j]) + ti.Vector([ox, oy])
        k1 = ti.Vector([qf_sample_value(vf_u, p, 0.0, 0.5),
                        qf_sample_value(vf_v, p, 0.5, 0.0)])
        p1 = p - 0.50 * _dt * k1

        k2 = ti.Vector([qf_sample_value(vf_u, p1, 0.0, 0.5),
                        qf_sample_value(vf_v, p1, 0.5, 0.0)])
        p2 = p - 0.75 * _dt * k2

        k3 = ti.Vector([qf_sample_value(vf_u, p2, 0.0, 0.5),
                        qf_sample_value(vf_v, p2, 0.5, 0.0)])

        p_prev = p - (_dt / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3)
        new_qf[i, j] = qf_sample_value(qf, p_prev, ox, oy)

@ti.kernel
def addInflow(qf: ti.template(), rect: ti.template(), value: ti.template()):
    lower_x, lower_y, upper_x, upper_y = rect[0], rect[1], rect[2], rect[3]
    for i, j in qf:
        if lower_x <= i <= upper_x and lower_y <= j <= upper_y:
            qf[i, j] = value

# it is kinda stupid that going to GPU with atomics is faster than python
@ti.kernel
def buffer_average(buf: ti.template()) -> ti.f32:
    avg = 0.0
    counter = 0.0
    for i, j in buf:
        avg += buf[i, j]
        counter += 1.0
    return avg / counter

# Slabs do the job of handling the double buffering nature of grid solvers
@ti.data_oriented
class Slab:
    def __init__(self, nx, ny, n_channels, outOfBoundsValue=0):
        if n_channels > 1:
            self.curr = ti.Vector.field(n_channels, float, shape=(nx, ny))
            self.next = ti.Vector.field(n_channels, float, shape=(nx, ny))
        else:
            self.curr = ti.field(float, shape=(nx, ny))
            self.next = ti.field(float, shape=(nx, ny))
        self.ox = 0.5
        self.oy = 0.5
        self.outOfBoundsValue = outOfBoundsValue

    def set_origin(self, ox, oy):
        self.ox = ox
        self.oy = oy

    def flip(self):
        self.curr, self.next = self.next, self.curr

    def advect(self, vf, _dt):
        advect_rk3(vf, self.curr, self.next, _dt)

    def advect_mac(self, vf_u, vf_v, _dt):
        advect_rk3_mac(vf_u, vf_v, self.curr, self.next, _dt, self.ox, self.oy)

    def set_inflow(self, rect, value):
        addInflow(self.curr, rect, value)

    def average(self):
        return buffer_average(self.curr)


################################################################################
# Boundary stuff
################################################################################

# apply boundary conditions to velocity field
@ti.kernel
def velocity_bc(vf: ti.template(), nx: int, ny: int):
    for i, j in vf:
        # corners
        if (i == j == 0) or (i == nx-1 and j == ny-1) or (i == 0 and j == ny-1) or (i == nx-1 and j == 0):
            vf[i,j] = ti.Vector([0.0, 0.0])
        # left wall
        elif i == 0:
            vf[i, j] = -vf[i+1,j]
        # bottom wall
        elif j == 0:
            vf[i, j] = -vf[i,j+1]
        # right wall
        elif i == nx-1:
            vf[i, j] = -vf[i-1,j]
        # top wall
        elif j == ny-1:
            vf[i, j] = -vf[i,j-1]

@ti.kernel
def velocity_bc_mac(vf: ti.template(), nx: int, ny: int):
    for i, j in vf:
        # corners
        if (i == j == 0) or (i == nx-1 and j == ny-1) or (i == 0 and j == ny-1) or (i == nx-1 and j == 0):
            vf[i,j] = 0.0
        # left wall
        elif i == 0:
            vf[i, j] = -vf[i+1,j]
        # bottom wall
        elif j == 0:
            vf[i, j] = -vf[i,j+1]
        # right wall
        elif i == nx-1:
            vf[i, j] = -vf[i-1,j]
        # top wall
        elif j == ny-1:
            vf[i, j] = -vf[i,j-1]


# apply boundary conditions to pressure field
@ti.kernel
def pressure_bc(pf: ti.template(), nx: int, ny: int):
    for i, j in pf:
        # corners
        if (i == j == 0) or (i == nx-1 and j == ny-1) or (i == 0 and j == ny-1) or (i == nx-1 and j == 0):
            pf[i,j] = 0.0
        # left wall
        elif i == 0:
            pf[i, j] = pf[i+1,j]
        # bottom wall
        elif j == 0:
            pf[i, j] = pf[i,j+1]
        # right wall
        elif i == nx-1:
            pf[i, j] = pf[i-1,j]
        # top wall
        elif j == ny-1:
            pf[i, j] = pf[i,j-1]

@ti.data_oriented
class BoundarySolver:
    def __init__(self):
        pass

    def setup_mac(self, vel_slab_u, vel_slab_v, pressure_slab):
        self.vel_slab_u = vel_slab_u
        self.vel_slab_v = vel_slab_v
        self.pressure_slab = pressure_slab
        self.is_mac = 1

    def setup(self, vel_slab, pressure_slab):
        self.vel_slab = vel_slab
        self.pressure_slab = pressure_slab
        self.is_mac = 0

    def update_velocity(self):
        if self.is_mac == 0:
            velocity_bc(self.vel_slab.curr, self.vel_slab.curr.shape[0],
                        self.vel_slab.curr.shape[1])
        else:
            velocity_bc_mac(self.vel_slab_u.curr, self.vel_slab_u.curr.shape[0],
                            self.vel_slab_u.curr.shape[1])
            velocity_bc_mac(self.vel_slab_v.curr, self.vel_slab_v.curr.shape[0],
                            self.vel_slab_v.curr.shape[1])

    def update_pressure(self):
        pressure_bc(self.pressure_slab.curr, self.pressure_slab.curr.shape[0],
                    self.pressure_slab.curr.shape[1])


@ti.kernel
def pressure_boundary_condition(pf: ti.template()):
    for i, j in pf:
        nx = pf.shape[0]
        ny = pf.shape[1]
        if (i == j == 0) or (i == nx-1 and j == ny-1) or (i == 0 and j == ny-1) or (i == nx-1 and j == 0):
            pf[i, j] = 0.0
        elif i == 0:
            pf[i, j] = pf[i+1, j]
        elif j == 0:
            pf[i, j] = pf[i, j+1]
        elif i == nx-1:
            pf[i, j] = pf[i-1, j]
        elif j == ny-1:
            pf[i, j] = pf[i, j-1]

@ti.func
def predict_p(pf, v_u, v_v, dt, i, j):
    dx = 0.5 * (qf_near_value(v_u, i+1, j) - qf_near_value(v_u, i, j))
    dy = 0.5 * (qf_near_value(v_v, i, j+1) - qf_near_value(v_v, i, j))
    pred = ((pf[i+1, j] + pf[i-1, j] + pf[i, j+1] + pf[i, j-1])
            - (dx + dy) / dt) * 0.25
    return pred


v2 = ti.types.vector(2, ti.f32)
@ti.data_oriented
class RedBlackSORSolver:
    def __init__(self, pressure_slab):
        self.pressure_slab = pressure_slab
        self.rel = 1.3

    def _update(self, p_next: ti.template(), p_curr: ti.template(),
                u_curr: ti.template(), v_curr: ti.template()) -> ti.f32:
        odd_res = self._update_odd(p_next, p_curr, u_curr, v_curr)
        even_res = self._update_even(p_next, p_curr, u_curr, v_curr)
        return ti.sqrt((odd_res[1] + even_res[1]) / (odd_res[0] + even_res[0]))

    @ti.func
    def _pn_ij(self, pc, uc, vc, i, j):
        return (1.0 - self.rel) * pc[i, j] + self.rel * predict_p(pc, uc, vc, self.dt, i, j)

    @ti.kernel
    def _update_odd(self, p_next: ti.template(), p_curr: ti.template(),
                    u_curr: ti.template(), v_curr: ti.template()) -> v2:
        norm_new = 0.0
        norm_diff = 0.0
        for i, j in p_next:
            if (i + j) % 2 == 1:
                nx = p_next.shape[0]
                ny = p_next.shape[1]
                if not (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                    p_next[i, j] = self._pn_ij(p_curr, u_curr, v_curr, i, j)
                    pf_diff = ti.abs(p_next[i, j] - p_curr[i, j])
                    norm_new += (p_next[i, j] * p_next[i, j])
                    norm_diff += (pf_diff * pf_diff)
        return v2([norm_new, norm_diff])

    @ti.kernel
    def _update_even(self, p_next: ti.template(), p_curr: ti.template(),
                     u_curr: ti.template(), v_curr: ti.template()) -> v2:
        norm_new = 0.0
        norm_diff = 0.0
        for i, j in p_next:
            if (i + j) % 2 == 0:
                nx = p_next.shape[0]
                ny = p_next.shape[1]
                if not (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                    p_next[i, j] = self._pn_ij(p_next, u_curr, v_curr, i, j)
                    pf_diff = ti.abs(p_next[i, j] - p_curr[i, j])
                    norm_new += (p_next[i, j] * p_next[i, j])
                    norm_diff += (pf_diff * pf_diff)
        return v2([norm_new, norm_diff])

    def update(self, u_slab, v_slab, dt):
        n_iters = 2
        self.dt = dt
        residual = 10.0
        for _ in range(n_iters):
            pressure_boundary_condition(self.pressure_slab.curr)
            residual = self._update(self.pressure_slab.next, self.pressure_slab.curr,
                                    u_slab.curr, v_slab.curr)
            self.pressure_slab.flip()
        pressure_boundary_condition(self.pressure_slab.curr)
        print("Error = " + str(residual))

@ti.data_oriented
class JacobiSolver:
    def __init__(self, pressure_slab):
        self.pressure_slab = pressure_slab

    @ti.kernel
    def _update(self, p_next: ti.template(), p_curr: ti.template(),
                u_curr: ti.template(), v_curr: ti.template()) -> ti.f32:
        norm_new = 0.0
        norm_diff = 0.0
        nx = p_next.shape[0]
        ny = p_next.shape[1]
        for i, j in p_next:
            if not (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                p_next[i, j] = predict_p(p_curr, u_curr, v_curr, self.dt, i, j)
                pf_diff = ti.abs(p_next[i, j] - p_curr[i, j])
                norm_new += (p_next[i, j] * p_next[i, j])
                norm_diff += (pf_diff * pf_diff)
        residual = ti.sqrt(norm_diff / norm_new)
        if norm_new == 0:
            residual = 0.0
        return residual

    def update(self, u_slab, v_slab, dt):
        n_iters = 10
        self.dt = dt
        residual = 10.0
        for _ in range(n_iters):
            pressure_boundary_condition(self.pressure_slab.curr)
            residual = self._update(self.pressure_slab.next, self.pressure_slab.curr,
                                    u_slab.curr, v_slab.curr)
            self.pressure_slab.flip()
        pressure_boundary_condition(self.pressure_slab.curr)
        print("Error = " + str(residual))
