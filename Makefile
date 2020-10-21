run_obj = src/bb2pbrt.o
mesh2p_obj = src/bbmesh2part.o
bubbles_obj = src/bubbles.o
core_objs = src/cuda/cutil.o src/core/kernel.o src/core/point_generator.o src/core/particle.o src/core/transform.o src/core/shape.o src/core/emitter.o src/core/collider.o src/core/statics.o

generator_objs = src/generator/triangle.o src/generator/bcclattice.o

third_objs = src/third/graphy.o src/third/obj_loader.o src/third/serializer.o

shape_objs = src/shapes/sphere2.o src/shapes/rectangle2.o src/shapes/bvh.o src/shapes/sphere.o src/shapes/box.o

solvers_objs = src/solvers/pcisph_solver2.o src/solvers/espic_solver2.o src/solvers/sph_solver2.o src/solvers/sph_gas_solver2.o src/solvers/sph_solver3.o src/solvers/pcisph_solver3.o

eqs_objs = src/equations/sph_equations2.o src/equations/espic_equations2.o src/equations/pcisph_equations2.o src/equations/sph_equations3.o src/equations/pcisph_equations3.o

test_objs = src/tests/test_grid.o src/tests/test_kernel.o src/tests/test_point_generator.o src/tests/test_math.o src/tests/test_sph2.o src/tests/test_espic.o src/tests/test_pcisph2.o src/tests/test_sph3.o src/tests/test_pcisph3.o

bb_objects = $(core_objs) $(bubbles_obj) $(shape_objs) $(generator_objs) $(solvers_objs) $(eqs_objs) $(third_objs) $(test_objs)
r_objects = $(core_objs) $(run_obj) $(shape_objs) $(generator_objs) $(solvers_objs) $(eqs_objs) $(third_objs) $(test_objs)
m2p_objects = $(core_objs) $(mesh2p_obj) $(shape_objs) $(generator_objs) $(solvers_objs) $(eqs_objs) $(third_objs) $(test_objs)

compile_debug_opts=-g -G
compile_fast_opts=-Xptxas -O3

compile_opts=$(compile_fast_opts) -gencode arch=compute_75,code=sm_75

includes = -I./src/cuda -I./src/core -I./src/third -I./src/third/graphy -I./src/tests -I./src/boundaries

libs = -ldl

all: bubbles bb2pbrt bbmesh2part

bbmesh2part: $(m2p_objects)
	nvcc $(compile_opts) $(m2p_objects) -o $@ $(libs)

bb2pbrt: $(r_objects)
	nvcc $(compile_opts) $(r_objects) -o $@ $(libs)

bubbles: $(bb_objects)
	nvcc $(compile_opts) $(bb_objects) -o $@ $(libs)

%.o: %.cpp
	nvcc -x cu $(compile_opts) $(includes) -dc $< -o $@

clean:
	rm -f src/*.o bubbles bb2pbrt src/core/*.o src/shapes/*.o src/cuda/*.o src/tests/*.o src/generator/*.o src/equations/*.o src/solvers/*.o src/third/*.o

rebuild: clean all
