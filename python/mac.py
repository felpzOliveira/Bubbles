import taichi as ti
import core.vmath
import core.grid
import core.exporter
import matplotlib.cm as cm
import solvers.smoke_solver2 as solvers2
import solvers.smoke_solver3 as solvers3
import solvers.smoke_solver_parameters as smoke_params

#ti.init(arch=ti.gpu, device_memory_GB=2)
ti.init(arch=ti.gpu)

res = 512
#res = 128
half_res = res / 2
c_len = 16 * res / 256
timestep = 0.005
#timestep = 0.01
pixel_mid = res // 2
ix_length = 15 * res / 512
iy_length = 10
domainX = 1.0

area = ti.Vector([pixel_mid - ix_length, 8, pixel_mid + ix_length, 8 + iy_length])

cube = ti.Vector([half_res-c_len, 2, half_res-c_len, half_res+c_len, c_len, half_res+c_len])
#cube1 = ti.Vector([2, 2, half_res-c_len, 8, c_len, half_res+c_len])
#cube2 = ti.Vector([res-10, 2, half_res-c_len, res-2, c_len, half_res+c_len])

#area = ti.Vector([8, pixel_mid - ix_length, 8 + iy_length, pixel_mid + ix_length])
inflow_velocity = ti.Vector([0.0, 20.0, 0.0])
inflow_density = 0.7

density = 1.0
dx = 1.0

params = smoke_params.SmokeSolverParameters()
#params.no_temperature()

solver = solvers2.SmokeSolver2D(res, res, params)
pSolver = core.grid.JacobiSolver(solver.pressure_slab)
#pSolver = core.grid.RedBlackSORSolver(solver.pressure_slab)

#solver = solvers3.SmokeSolver3D(res, res, res, params)
#pSolver = core.grid.JacobiSolver3D(solver.pressure_slab)
#pSolver = core.grid.RedBlackSORSolver3D(solver.pressure_slab)

solver.set_pressure_solver(pSolver)
gui = ti.GUI('Basic Smoke Solver', (res, res))
step = 0
emit = True

while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: exit(0)
        if gui.event.key == 'a':
            emit = not emit
        if gui.event.key == 's':
            if solver.dimensions() == 3:
                ds = domainX / float(res)
                core.exporter.as_vol(solver.density_slab.curr,
                                     "some_file.vol", [ds, ds, ds])
    for itr in range(10):
        step += 1
        # Add inflow:
        if emit:
            if solver.dimensions() == 3:
                solver.handle_inflow(cube, inflow_velocity, inflow_density)
            else:
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
        solver.solve_pressure(timestep)

        solver.apply_pressure(timestep, density, dx)
        solver.flip_velocities()

    #print("Def = " + str(step))
    solver.flat_image()
    gui.set_image(solver.pixels())
    gui.show()

