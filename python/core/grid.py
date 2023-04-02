import taichi as ti
import core.vmath
import numpy as np

################################################################################
# Buffer manipulation
################################################################################

# return the value of quantity field 'qf' at location u, v, clamped if required
@ti.func
def qf_near_value2D(qf: ti.template(), u, v):
    i = max(0, min(int(u), qf.shape[0]-1))
    j = max(0, min(int(v), qf.shape[1]-1))
    return qf[i, j]

@ti.func
def qf_near_value3D(qf: ti.template(), u, v, w):
    i = max(0, min(int(u), qf.shape[0]-1))
    j = max(0, min(int(v), qf.shape[1]-1))
    k = max(0, min(int(w), qf.shape[2]-1))
    return qf[i, j, k]

# samples a value of quantity field 'qf' at coordinate p = (u,v)
# note that this is done in grid coordinates where grid spacing is dx = 1
# ox, oy account for where is the measure made in 'qf', i.e.:
# for regular grids values are measured in the center with ox = oy = 0.5
# for mac grids these differ for the u component ox = 0.0, oy = 0.5
# and v component ox = 0.5, oy = 0.0
@ti.func
def qf_sample_value2D(qf: ti.template(), p: ti.template(),
                      ox: float, oy: float):
    s, t = p[0] - ox, p[1] - oy
    iu, iv = int(s), int(t)
    fu, fv = s - iu, t - iv
    f00 = qf_near_value2D(qf, iu + 0, iv + 0)
    f10 = qf_near_value2D(qf, iu + 1, iv + 0)
    f01 = qf_near_value2D(qf, iu + 0, iv + 1)
    f11 = qf_near_value2D(qf, iu + 1, iv + 1)
    return core.vmath.bilerp(f00, f10, f01, f11, fu, fv)

@ti.func
def qf_sample_value3D(qf: ti.template(), p: ti.template(),
                      ox : float, oy: float, oz : float):
    s, t, w = p[0] - ox, p[1] - oy, p[2] - oz
    iu, iv, iw = int(s), int(t), int(w)
    fu, fv, fw = s - iu, t - iv, w - iw
    f000 = qf_near_value3D(qf, iu + 0, iv + 0, iw + 0)
    f100 = qf_near_value3D(qf, iu + 1, iv + 0, iw + 0)
    f010 = qf_near_value3D(qf, iu + 0, iv + 1, iw + 0)
    f110 = qf_near_value3D(qf, iu + 1, iv + 1, iw + 0)
    f001 = qf_near_value3D(qf, iu + 0, iv + 0, iw + 1)
    f101 = qf_near_value3D(qf, iu + 1, iv + 0, iw + 1)
    f011 = qf_near_value3D(qf, iu + 0, iv + 1, iw + 1)
    f111 = qf_near_value3D(qf, iu + 1, iv + 1, iw + 1)
    return core.vmath.trilerp(f000, f100, f010, f110, f001, f101, f011, f111, fu, fv, fw)

# do advection over the velocity field described by explicit components 'u' and 'v'
# considering the offset of data points in the layout of the MAC grid. Quantity 'qf'
# is allowed to have any offset 'ox', 'oy' in the range [0.0, 1.0]. Note that this
# routine (like others) work on grid coordinates and not world coordinates, it is
# assumed that grids are on top of each other and no translation/scaling is present
@ti.kernel
def advect_rk3_2D(vf_u: ti.template(), vf_v: ti.template(),
                  qf: ti.template(), new_qf: ti.template(), _dt: float,
                  ox: float, oy: float):
    for i,j in qf:
        p = ti.Vector([i, j]) + ti.Vector([ox, oy])
        k1 = ti.Vector([qf_sample_value2D(vf_u, p, 0.0, 0.5),
                        qf_sample_value2D(vf_v, p, 0.5, 0.0)])
        p1 = p - 0.50 * _dt * k1

        k2 = ti.Vector([qf_sample_value2D(vf_u, p1, 0.0, 0.5),
                        qf_sample_value2D(vf_v, p1, 0.5, 0.0)])
        p2 = p - 0.75 * _dt * k2

        k3 = ti.Vector([qf_sample_value2D(vf_u, p2, 0.0, 0.5),
                        qf_sample_value2D(vf_v, p2, 0.5, 0.0)])

        p_prev = p - (_dt / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3)
        new_qf[i, j] = qf_sample_value2D(qf, p_prev, ox, oy)

@ti.kernel
def advect_rk3_3D(vf_u: ti.template(), vf_v: ti.template(), vf_w: ti.template(),
                  qf: ti.template(), new_qf: ti.template(), _dt: float,
                  ox: float, oy: float, oz: float):
    for i,j,k in qf:
        p = ti.Vector([i, j, k]) + ti.Vector([ox, oy, oz])
        k1 = ti.Vector([qf_sample_value3D(vf_u, p, 0.0, 0.5, 0.5),
                        qf_sample_value3D(vf_v, p, 0.5, 0.0, 0.5),
                        qf_sample_value3D(vf_w, p, 0.5, 0.5, 0.0)])
        p1 = p - 0.50 * _dt * k1

        k2 = ti.Vector([qf_sample_value3D(vf_u, p1, 0.0, 0.5, 0.5),
                        qf_sample_value3D(vf_v, p1, 0.5, 0.0, 0.5),
                        qf_sample_value3D(vf_w, p1, 0.5, 0.5, 0.0)])
        p2 = p - 0.75 * _dt * k2

        k3 = ti.Vector([qf_sample_value3D(vf_u, p2, 0.0, 0.5, 0.5),
                        qf_sample_value3D(vf_v, p2, 0.5, 0.0, 0.5),
                        qf_sample_value3D(vf_w, p2, 0.5, 0.5, 0.0)])

        p_prev = p - (_dt / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3)
        new_qf[i, j, k] = qf_sample_value3D(qf, p_prev, ox, oy, oz)


@ti.kernel
def addInflow2D(qf: ti.template(), rect: ti.template(), value: ti.template()):
    lower_x, lower_y, upper_x, upper_y = rect[0], rect[1], rect[2], rect[3]
    for i, j in qf:
        if lower_x <= i <= upper_x and lower_y <= j <= upper_y:
            qf[i, j] = value

@ti.kernel
def addInflow3D(qf: ti.template(), cube: ti.template(), value: ti.template()):
    lower_x, lower_y, lower_z = cube[0], cube[1], cube[2]
    upper_x, upper_y, upper_z = cube[3], cube[4], cube[5]
    for i, j, k in qf:
        if lower_x <= i <= upper_x and lower_y <= j <= upper_y and lower_z <= k <= upper_z:
            qf[i, j, k] = value

# it is kinda stupid that going to GPU with atomics is faster than python
@ti.kernel
def buffer_average2D(buf: ti.template()) -> ti.f32:
    avg = 0.0
    counter = 0.0
    for i, j in buf:
        avg += buf[i, j]
        counter += 1.0
    return avg / counter

@ti.kernel
def buffer_average3D(buf: ti.template()) -> ti.f32:
    avg = 0.0
    counter = 0.0
    for i, j, k in buf:
        avg += buf[i, j, k]
        counter += 1.0
    return avg / counter

# Slabs do the job of handling the double buffering nature of grid solvers
@ti.data_oriented
class Slab2D:
    def __init__(self, nx, ny, n_channels):
        if n_channels > 1:
            self.curr = ti.Vector.field(n_channels, float, shape=(nx, ny))
            self.next = ti.Vector.field(n_channels, float, shape=(nx, ny))
        else:
            self.curr = ti.field(float, shape=(nx, ny))
            self.next = ti.field(float, shape=(nx, ny))
        self.ox = 0.5
        self.oy = 0.5

    def set_origin(self, ox, oy):
        self.ox = ox
        self.oy = oy

    def flip(self):
        self.curr, self.next = self.next, self.curr

    def advect(self, vf_u, vf_v, _dt):
        advect_rk3_2D(vf_u, vf_v, self.curr, self.next, _dt, self.ox, self.oy)

    def set_inflow(self, rect, value):
        addInflow2D(self.curr, rect, value)

    def average(self):
        return buffer_average2D(self.curr)

@ti.data_oriented
class Slab3D:
    def __init__(self, nx, ny, nz, n_channels):
        if n_channels > 1:
            self.curr = ti.Vector.field(n_channels, float, shape=(nx, ny, nz))
            self.next = ti.Vector.field(n_channels, float, shape=(nx, ny, nz))
        else:
            self.curr = ti.field(float, shape=(nx, ny, nz))
            self.next = ti.field(float, shape=(nx, ny, nz))
        self.ox = 0.5
        self.oy = 0.5
        self.oz = 0.5

    def set_origin(self, ox, oy, oz):
        self.ox = ox
        self.oy = oy
        self.oz = oz

    def flip(self):
        self.curr, self.next = self.next, self.curr

    def advect(self, vf_u, vf_v, vf_w, _dt):
        advect_rk3_3D(vf_u, vf_v, vf_w, self.curr, self.next, _dt,
                      self.ox, self.oy, self.oz)

    def set_inflow(self, cube, value):
        addInflow3D(self.curr, cube, value)

    def average(self):
        return buffer_average3D(self.curr)

################################################################################
# Boundary stuff
################################################################################

@ti.kernel
def velocity_bc2D(vf: ti.template(), nx: int, ny: int):
    for i, j in vf:
        # corners, bottom
        if (i == j == 0) or (i == nx-1 and j == 0):
            vf[i,j] = 0.0
        # corners, top
        elif (i == 0 and j == ny-1) or (i == nx-1 and j == ny-1):
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

@ti.kernel
def velocity_bc3D(vf: ti.template(), nx: int, ny: int, nz: int):
    for i, j, k in vf:
        # corners, bottom of the cube
        if (i == j == k == 0) or (i == nx-1 and j == k == 0) or (
            i == nx-1 and j == 0 and k == nz-1) or (i == j == 0 and k == nz-1):
            vf[i,j,k] = 0.0
        # corners, top of the cube
        elif (i == k == 0 and j == ny-1) or (i == nx-1 and j == ny-1 and k == 0) or (
              i == nx-1 and j == ny-1 and k == nz-1) or (i == 0 and j == ny-1 and k == nz-1):
            vf[i,j,k] = 0.0
        # horizontal face at x = 0
        elif i == 0:
            vf[i,j,k] = -vf[i+1,j,k]
        # vertical face at y = 0
        elif j == 0:
            vf[i,j,k] = -vf[i,j+1,k]
        # horizontal face at x = nx-1
        elif i == nx-1:
            vf[i,j,k] = -vf[i-1,j,k]
        # vertical face at y = ny-1
        elif j == ny-1:
            vf[i,j,k] = -vf[i,j-1,k]
        # depth face at z = 0
        elif k == 0:
            vf[i,j,k] = -vf[i,j,k+1]
        # depth face at z = nz-1
        elif k == nz-1:
            vf[i,j,k] = -vf[i,j,k-1]

def create_bc_array(nx, ny):
    return np.zeros((nx, ny), dtype=np.uint8)

def set_plane(bc, lower, upper):
    bc[lower[0] : upper[0], lower[1] : upper[1]] = 1

def set_circle(bc, center, radius):
    center = np.asarray(center)
    l_ = np.round(np.maximum(center - radius, 0)).astype(np.int32)
    u0 = round(min(center[0] + radius, bc.shape[0]))
    u1 = round(min(center[1] + radius, bc.shape[1]))
    for i in range(l_[0], u0):
        for j in range(l_[1], u1):
            x = np.array([i, j]) + 0.5
            if np.linalg.norm(x - center) < radius:
                bc[i, j] = 1

@ti.data_oriented
class BoundarySolver2D:
    def __init__(self):
        pass

    @staticmethod
    def to_field(bc):
        bc_field = ti.field(ti.u8, shape=bc.shape[:2])
        bc_field.from_numpy(bc)
        return bc_field

    def setup(self, vel_slab_u, vel_slab_v):
        self.vel_slab_u = vel_slab_u
        self.vel_slab_v = vel_slab_v
        nx = self.vel_slab_u.curr.shape[0]-1
        ny = self.vel_slab_v.curr.shape[1]-1
        bc = create_bc_array(nx, ny)
        set_plane(bc, (0, 0), (nx, 2))
        set_plane(bc, (0,ny-2), (nx, ny))
        set_plane(bc, (nx-2,0), (nx, ny))
        set_plane(bc, (nx-2,0), (nx, ny))
        set_plane(bc, (0,0), (2, ny))
        #set_circle(bc, (256, 256), 30)
        self.bc = BoundarySolver2D.to_field(bc)

    @ti.func
    def mask(self, i: int, j : int):
        rv = 0
        if i < 0 or j < 0 or i >= self.bc.shape[0] or j >= self.bc.shape[1]:
            rv = 1
        else:
            rv = self.bc[i, j]
        return rv

    @ti.func
    def is_wall(self, i : int, j : int):
        rv = False
        if i < 0 or j < 0 or i >= self.bc.shape[0] or j >= self.bc.shape[1]:
            rv = True
        else:
            rv = self.bc[i, j] == 1
        return rv

    @ti.kernel
    def set_velocity_boundary(self, vf: ti.template()):
        bc = ti.static(self.bc)
        for i, j in vf:
            if bc[i, j] == 1:
                if self.mask(i-1,j) == 0 and self.mask(i,j-1) == 1 and self.mask(i,j+1) == 1:
                    vf[i+1,j] = -vf[i-1,j]
                elif self.mask(i+1,j) == 0 and self.mask(i,j-1) == 1 and self.mask(i,j+1) == 1:
                    vf[i-1,j] = -vf[i+1,j]
                elif self.mask(i,j-1) == 0 and self.mask(i-1,j) == 1 and self.mask(i+1,j) == 1:
                    vf[i,j+1] = -vf[i,j-1]
                elif self.mask(i,j+1) == 0 and self.mask(i-1,j) == 1 and self.mask(i+1,j) == 1:
                    vf[i, j-1] = -vf[i, j+1]

    @ti.kernel
    def set_pressure_boundary(self, pc: ti.template()):
        bc = ti.static(self.bc)
        for i, j in pc:
            if bc[i, j] == 1:
                if self.mask(i-1,j) == 0 and self.mask(i,j-1) == 1 and self.mask(i,j+1) == 1:
                    pc[i, j] = pc[i - 1, j]
                elif self.mask(i+1,j) == 0 and self.mask(i,j-1) == 1 and self.mask(i,j+1) == 1:
                    pc[i, j] = pc[i + 1, j]
                elif self.mask(i,j-1) == 0 and self.mask(i-1,j) == 1 and self.mask(i+1,j) == 1:
                    pc[i, j] = pc[i, j - 1]
                elif self.mask(i,j+1) == 0 and self.mask(i-1,j) == 1 and self.mask(i+1,j) == 1:
                    pc[i, j] = pc[i, j + 1]
                elif self.mask(i-1,j) == 0 and self.mask(i,j+1) == 0:
                    pc[i, j] = (pc[i - 1, j] + pc[i, j + 1]) / 2.0
                elif self.mask(i+1,j) == 0 and self.mask(i,j+1) == 0:
                    pc[i, j] = (pc[i + 1, j] + pc[i, j + 1]) / 2.0
                elif self.mask(i-1,j) == 0 and self.mask(i,j-1) == 0:
                    pc[i, j] = (pc[i - 1, j] + pc[i, j - 1]) / 2.0
                elif self.mask(i+1,j) == 0 and self.mask(i,j-1) == 0:
                    pc[i, j] = (pc[i + 1, j] + pc[i, j - 1]) / 2.0

    def update_velocity(self):
        self.set_velocity_boundary(self.vel_slab_u.curr)
        self.set_velocity_boundary(self.vel_slab_v.curr)
        #velocity_bc2D(self.vel_slab_u.curr, self.vel_slab_u.curr.shape[0],
         #             self.vel_slab_u.curr.shape[1])
        #velocity_bc2D(self.vel_slab_v.curr, self.vel_slab_v.curr.shape[0],
         #             self.vel_slab_v.curr.shape[1])

@ti.data_oriented
class BoundarySolver3D:
    def __init__(self):
        pass

    def setup(self, vel_slab_u, vel_slab_v, vel_slab_w):
        self.vel_slab_u = vel_slab_u
        self.vel_slab_v = vel_slab_v
        self.vel_slab_w = vel_slab_w

    def update_velocity(self):
        velocity_bc3D(self.vel_slab_u.curr, self.vel_slab_u.curr.shape[0],
                      self.vel_slab_u.curr.shape[1], self.vel_slab_u.curr.shape[2])
        velocity_bc3D(self.vel_slab_v.curr, self.vel_slab_v.curr.shape[0],
                      self.vel_slab_v.curr.shape[1], self.vel_slab_v.curr.shape[2])
        velocity_bc3D(self.vel_slab_w.curr, self.vel_slab_w.curr.shape[0],
                      self.vel_slab_w.curr.shape[1], self.vel_slab_w.curr.shape[2])


################################################################################
# Pressure stuff
################################################################################

@ti.kernel
def pressure_boundary_condition(pf: ti.template()):
    for i, j in pf:
        nx = pf.shape[0]
        ny = pf.shape[1]
        if (i == j == 0) or (i == nx-1 and j == ny-1) or (
            i == 0 and j == ny-1) or (i == nx-1 and j == 0):
            pf[i, j] = 0.0
        elif i == 0:
            pf[i, j] = pf[i+1, j]
        elif j == 0:
            pf[i, j] = pf[i, j+1]
        elif i == nx-1:
            pf[i, j] = pf[i-1, j]
        elif j == ny-1:
            pf[i, j] = pf[i, j-1]

@ti.kernel
def pressure_boundary_condition3(pf: ti.template()):
    for i, j, k in pf:
        nx = pf.shape[0]
        ny = pf.shape[1]
        nz = pf.shape[2]
        # corners, bottom of the cube
        if (i == j == k == 0) or (i == nx-1 and j == k == 0) or (
            i == nx-1 and j == 0 and k == nz-1) or (i == j == 0 and k == nz-1):
            pf[i,j,k] = 0.0
        # corners, top of the cube
        elif (i == k == 0 and j == ny-1) or (i == nx-1 and j == ny-1 and k == 0) or (
              i == nx-1 and j == ny-1 and k == nz-1) or (i == 0 and j == ny-1 and k == nz-1):
            pf[i,j,k] = 0.0
        elif i == 0:
            pf[i,j,k] = pf[i+1,j,k]
        elif j == 0:
            pf[i,j,k] = pf[i,j+1,k]
        elif i == nx-1:
            pf[i,j,k] = pf[i-1,j,k]
        elif j == ny-1:
            pf[i,j,k] = pf[i,j-1,k]
        elif k == nz-1:
            pf[i,j,k] = pf[i,j,k-1]
        elif k == 0:
            pf[i,j,k] = pf[i,j,k+1]

# Compute the pressure based on the velocity field and the base stencil, i.e.:
# on the MAC grid:
#        |            |
#        |    Pi,j+1  |
#________|___Vi+1,j___|____________
#        |            |
#    Ui,j|    Pi,j    |Ui+1,j
# Pi-1,j |            |    Pi+1,j
#________|____________|____________
#        |   Vi,j     |
#        |    Pi,j-1  |
#        |            |
#
# pressure at index (i, j) is given by the *pressure* at (i+1,j), (i-1,j),
# (i,j+1) and (i,j-1), the *velocities U/V* at the faces, i.e.:
# (i,j), (i+1,j), (i,j+1), so we have:
#
#        Pi,j ~ (Pi+1,j + Pi-1,j + Pi,j+1 + Pi,j-1)
# The usual jacobi iteration is then:
#    x(k+1) = (1/aii) (bi - Σ aij x(k))
# the matrix A has aii = 4 in 2D and bi = Divi,j
# Because we sample velocities at faces we have:
#    Divi,j = ((Ui+1,j - Ui,j) + (Vi,j+1, - Vi,j)) * 0.5
# so we can simply write:
#    Pi,j(k) = ((Pi+1,j + Pi-1,j + Pi,j+1 + Pi,j-1) - (Divi,j)) * 0.25
#
# For 3D we have to consider the cells at (i,j,k+1) and (i,j,k-1) so the amount of
# cells increase to 6, and therefore we simply add the pressure on these 2 cells
# and the divergence is increased by the z differentiation, all multiplied by 1/6
# instead of 1/4.
@ti.func
def p_at_cell(pf, v_u, v_v, dt, i, j):
    term = (1.0 / 0.1) * 1.25
    dx = (qf_near_value2D(v_u, i+1, j) - qf_near_value2D(v_u, i, j))
    dy = (qf_near_value2D(v_v, i, j+1) - qf_near_value2D(v_v, i, j))
    return ((pf[i+1, j] + pf[i-1, j] + pf[i, j+1] + pf[i, j-1]) - (dx + dy) * term / dt * (0.1 ** 2)) * 0.25

@ti.func
def p_at_cell3(pf, v_u, v_v, v_w, dt, i, j, k):
    dx = 0.5 * (qf_near_value3D(v_u, i+1, j, k) - qf_near_value3D(v_u, i, j, k))
    dy = 0.5 * (qf_near_value3D(v_v, i, j+1, k) - qf_near_value3D(v_v, i, j, k))
    dz = 0.5 * (qf_near_value3D(v_w, i, j, k+1) - qf_near_value3D(v_w, i, j, k))
    return ((pf[i+1,j,k] + pf[i-1,j,k] + pf[i,j+1,k] + pf[i,j-1,k] + pf[i,j,k+1] + pf[i,j,k-1]) - (dx + dy + dz) / dt) * 0.166666

v2 = ti.types.vector(2, ti.f32)
@ti.data_oriented
class RedBlackSORSolver:
    def __init__(self, pressure_slab):
        self.pressure_slab = pressure_slab
        self.rel = 1.3

    def set_boundary_solver(self, bSolver):
        self.bSolver = bSolver

    def _update(self, p_next: ti.template(), p_curr: ti.template(),
                u_curr: ti.template(), v_curr: ti.template()) -> ti.f32:
        odd_res = self._update_odd(p_next, p_curr, u_curr, v_curr)
        even_res = self._update_even(p_next, p_curr, u_curr, v_curr)
        return ti.sqrt((odd_res[1] + even_res[1]) / (odd_res[0] + even_res[0]))

    @ti.func
    def compute_pij(self, pc, uc, vc, i, j):
        return (1.0 - self.rel) * pc[i, j] + self.rel * p_at_cell(pc, uc, vc, self.dt, i, j)

    @ti.kernel
    def _update_odd(self, p_next: ti.template(), p_curr: ti.template(),
                    u_curr: ti.template(), v_curr: ti.template()) -> v2:
        norm_new = 0.0
        norm_diff = 0.0
        for i, j in p_next:
            if (i + j) % 2 == 1:
                nx = p_next.shape[0]
                ny = p_next.shape[1]
                if not self.bSolver.is_wall(i, j):
                #if not (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                    p_next[i, j] = self.compute_pij(p_curr, u_curr, v_curr, i, j)
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
                if not self.bSolver.is_wall(i, j):
                #if not (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                    p_next[i, j] = self.compute_pij(p_next, u_curr, v_curr, i, j)
                    pf_diff = ti.abs(p_next[i, j] - p_curr[i, j])
                    norm_new += (p_next[i, j] * p_next[i, j])
                    norm_diff += (pf_diff * pf_diff)
        return v2([norm_new, norm_diff])

    def update(self, u_slab, v_slab, dt):
        n_iters = 2
        self.dt = dt
        residual = 10.0
        for _ in range(n_iters):
            #pressure_boundary_condition(self.pressure_slab.curr)
            self.bSolver.set_pressure_boundary(self.pressure_slab.curr)
            residual = self._update(self.pressure_slab.next, self.pressure_slab.curr,
                                    u_slab.curr, v_slab.curr)
            self.pressure_slab.flip()
        #pressure_boundary_condition(self.pressure_slab.curr)
        self.bSolver.set_pressure_boundary(self.pressure_slab.curr)
        print("Error = " + str(residual))

@ti.data_oriented
class JacobiSolver:
    def __init__(self, pressure_slab):
        self.pressure_slab = pressure_slab

    def set_boundary_solver(self, bSolver):
        self.bSolver = bSolver

    @ti.kernel
    def _update(self, p_next: ti.template(), p_curr: ti.template(),
                u_curr: ti.template(), v_curr: ti.template()) -> ti.f32:
        norm_new = 0.0
        norm_diff = 0.0
        nx = p_next.shape[0]
        ny = p_next.shape[1]
        for i, j in p_next:
            if not self.bSolver.is_wall(i, j):
            #if not (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                p_next[i, j] = p_at_cell(p_curr, u_curr, v_curr, self.dt, i, j)
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
            #pressure_boundary_condition(self.pressure_slab.curr)
            self.bSolver.set_pressure_boundary(self.pressure_slab.curr)
            residual = self._update(self.pressure_slab.next, self.pressure_slab.curr,
                                    u_slab.curr, v_slab.curr)
            self.pressure_slab.flip()
        #pressure_boundary_condition(self.pressure_slab.curr)
        self.bSolver.set_pressure_boundary(self.pressure_slab.curr)
        print("Error = " + str(residual))

@ti.data_oriented
class RedBlackSORSolver3D:
    def __init__(self, pressure_slab):
        self.pressure_slab = pressure_slab
        self.rel = 1.3

    def _update(self, p_next: ti.template(), p_curr: ti.template(), u_curr: ti.template(),
                v_curr: ti.template(), w_curr: ti.template()) -> ti.f32:
        odd_res = self._update_odd(p_next, p_curr, u_curr, v_curr, w_curr)
        even_res = self._update_even(p_next, p_curr, u_curr, v_curr, w_curr)
        return ti.sqrt((odd_res[1] + even_res[1]) / (odd_res[0] + even_res[0]))

    @ti.func
    def compute_pijk(self, pc, uc, vc, wc, i, j, k):
        return (1.0 - self.rel) * pc[i,j,k] + self.rel * p_at_cell3(pc, uc, vc, wc, self.dt,i,j,k)

    @ti.kernel
    def _update_odd(self, p_next: ti.template(), p_curr: ti.template(), u_curr: ti.template(),
                    v_curr: ti.template(), w_curr: ti.template()) -> v2:
        norm_new = 0.0
        norm_diff = 0.0
        for i, j, k in p_next:
            if (i + j + k) % 2 == 1: # TODO: This looks sus
                nx = p_next.shape[0]
                ny = p_next.shape[1]
                nz = p_next.shape[2]
                if not (i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1):
                    p_next[i,j,k] = self.compute_pijk(p_curr,u_curr,v_curr,w_curr,i,j,k)
                    pf_diff = ti.abs(p_next[i, j, k] - p_curr[i, j, k])
                    norm_new += (p_next[i, j, k] * p_next[i, j, k])
                    norm_diff += (pf_diff * pf_diff)
        return v2([norm_new, norm_diff])

    @ti.kernel
    def _update_even(self, p_next: ti.template(), p_curr: ti.template(),
                     u_curr: ti.template(), v_curr: ti.template(), w_curr: ti.template()) -> v2:
        norm_new = 0.0
        norm_diff = 0.0
        for i, j, k in p_next:
            if (i + j + k) % 2 == 0:
                nx = p_next.shape[0]
                ny = p_next.shape[1]
                nz = p_next.shape[2]
                if not (i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1):
                    p_next[i,j,k] = self.compute_pijk(p_next,u_curr,v_curr,w_curr,i,j,k)
                    pf_diff = ti.abs(p_next[i, j, k] - p_curr[i, j, k])
                    norm_new += (p_next[i, j, k] * p_next[i, j, k])
                    norm_diff += (pf_diff * pf_diff)
        return v2([norm_new, norm_diff])

    def update(self, u_slab, v_slab, w_slab, dt):
        n_iters = 2
        self.dt = dt
        residual = 10.0
        for _ in range(n_iters):
            pressure_boundary_condition3(self.pressure_slab.curr)
            residual = self._update(self.pressure_slab.next, self.pressure_slab.curr,
                                    u_slab.curr, v_slab.curr, w_slab.curr)
            self.pressure_slab.flip()
        pressure_boundary_condition3(self.pressure_slab.curr)
        print("Error = " + str(residual))

@ti.data_oriented
class JacobiSolver3D:
    def __init__(self, pressure_slab):
        self.pressure_slab = pressure_slab

    @ti.kernel
    def _update(self, p_next: ti.template(), p_curr: ti.template(), u_curr: ti.template(),
                v_curr: ti.template(), w_curr: ti.template()) -> ti.f32:
        norm_new = 0.0
        norm_diff = 0.0
        nx = p_next.shape[0]
        ny = p_next.shape[1]
        nz = p_next.shape[2]
        for i, j, k in p_next:
            if not (i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1):
                p_next[i,j,k] = p_at_cell3(p_curr, u_curr, v_curr, w_curr, self.dt, i, j, k)
                pf_diff = ti.abs(p_next[i,j,k] - p_curr[i,j,k])
                norm_new += (p_next[i,j,k] * p_next[i,j,k])
                norm_diff += (pf_diff * pf_diff)
        residual = ti.sqrt(norm_diff / norm_new)
        if norm_new == 0:
            residual = 0.0
        return residual

    def update(self, u_slab, v_slab, w_slab, dt):
        n_iters = 120
        self.dt = dt
        residual = 10.0
        for _ in range(n_iters):
            pressure_boundary_condition3(self.pressure_slab.curr)
            residual = self._update(self.pressure_slab.next, self.pressure_slab.curr,
                                    u_slab.curr, v_slab.curr, w_slab.curr)
            self.pressure_slab.flip()
        pressure_boundary_condition3(self.pressure_slab.curr)
        print("Error = " + str(residual))
