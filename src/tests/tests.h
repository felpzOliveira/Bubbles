#pragma once
#include <cutil.h>
#include <geometry.h>
#include <sph_solver.h>

#define TEST_CHECK(x, msg) test_check((x), #x, __FILE__, __LINE__, msg)

/*
* I'm not gonna write a test framework *again*, I'm writting test functions
* and calling them in a separate functions, deal with it.
*/

inline __host__
void test_check(bool v, const char *name, const char *filename, int line, const char *msg){
    if(!v){
        if(!msg)
            printf("Check: %s (%s:%d) : (No message)\n", name, filename, line);
        else
            printf("Check: %s (%s:%d) : (%s)\n", name, filename, line, msg);

        cudaSafeExit();
    }
}

vec3f get_color_level(int level);

// Utilities for coloring
template<typename F> inline int
set_poscol_lnm(float *pos, float *col, ParticleSet3 *pSet, Grid3 *grid, F accept){
    int it = 0;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec3f pi = pSet->GetParticlePosition(i);
        if(accept(pi)){
            unsigned int id = grid->GetLinearHashedPosition(pi);
            Cell3 *cell = grid->GetCell(id);
            vec3f color = get_color_level(cell->GetLevel());
            pos[3 * it + 0] = pi.x; pos[3 * it + 1] = pi.y;
            pos[3 * it + 2] = pi.z;
            col[3 * it + 0] = color[0]; col[3 * it + 1] = color[1];
            col[3 * it + 2] = color[2];
            it ++;
        }
    }

    return it;
}


void simple_color(float *pos, float *col, ParticleSet3 *pSet);
void update_colors_lnm(float *col, SphSolverData2 *data);
void set_colors_lnm(float *col, SphSolverData2 *data, int is_first=1, int classify=1);
int  set_poscol_lnm(float *col, float *pos, SphSolverData3 *data, int is_first=0, int classify=1);
void set_colors_temperature(float *col, SphSolverData2 *data);
void set_colors_pressure(float *col, SphSolverData2 *data);
void graphy_vector_set(vec3f origin, vec3f target, Float fov, Float near, Float far);
void graphy_vector_set(vec3f origin, vec3f target);


void test_uniform_grid2D();
void test_uniform_grid3D();
void test_distribute_uniform_grid2D();
void test_distribute_uniform_grid3D();
void test_distribute_random_grid2D();
void test_distribute_random_grid3D();
void test_neighbor_query_grid2D();
void test_neighbor_query_grid3D();
void test_node_grid_2D();
void test_particle_to_node_2D();
void test_grid_face_centered_2D();
void test_grid_face_centered_3D();
void test_explicit_grid_build_2D();
void test_explicit_grid_minimal_build_2D();
void test_explicit_vector_grid_build_2D();
void test_explicit_face_vector_grid_2D();

void test_kernels_2D();
void test_kernels_3D();

void test_simple_triangle_distance();
void test_triangle_point_generator();
void test_bcclattice_point_generator();
void test_volume_particle_emitter3();
void test_volume_particle_emitter3_mesh();
void test_mesh_collision();
void test_continuous_builder2D();
void test_matrix_operations();
void test_matrix_operations2();
void test_sampling_barycentric();
void test_ray2_intersect();
void test_closest_point_sphere2D();
void test_color_field_2D();
void test_rectangle_distance_outside();
void test_rectangle_distance_inside();
void test_box_distance();
void test_rectangle_emit();
void test_container_grid_sdf_2D();
void test_bounds_split2();

void test_sph2_water_block();
void test_sph2_water_sphere();
void test_sph2_double_dam_break();
void test_sph2_water_drop();
void test_sph2_gas_sphere();

void test_sph3_water_sphere();
void test_sph3_water_block();
void test_sph3_double_dam_break();

void test_pcisph2_water_block();
void test_pcisph2_water_sphere();
void test_pcisph2_double_dam_break();
void test_pcisph2_water_drop();
void test_pcisph2_water_block_lnm();
void test_pcisph2_continuous_emitter();
void test_pcisph2_marching_squares();
void test_pcisph2_water_rotating_obstacles();
void test_pcisph2_water_sphere_dynamic();
void test_pcisph2_water_square_dynamic();

// As you can see, I'm having a lot fun with the pcisph3 solver
void test_pcisph3_water_sphere();
void test_pcisph3_double_dam_break();
void test_pcisph3_dragon();
void test_pcisph3_lucy_dam();
void test_pcisph3_happy_whale();
void test_pcisph3_whale_obstacle();
void test_pcisph3_lucy_ball();
void test_pcisph3_quadruple_dam();
void test_pcisph3_multiple_emission();
void test_pcisph3_dragon_shower();
void test_pcisph3_ball_many_emission();
void test_pcisph3_box_many_emission();
void test_pcisph3_water_drop();
void test_pcisph3_dragon_pool();
void test_pcisph3_water_sphere_movable();
void test_pcisph3_sdf();
void test_pcisph3_rotating_water_box();
void test_pcisph3_water_box_forward();
void test_pcisph3_dam_break_double_dragon();
void test_pcisph3_box_mesh();

void test_pbf2_double_dam_break();

void test_lnm_happy_whale();

void test_espic_1D();
void test_espic_particles_2D();

void test_field_grid();
