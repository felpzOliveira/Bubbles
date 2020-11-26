#include <cutil.h>
#include <transform.h>
#include <particle.h>
#include <emitter.h>
#include <obj_loader.h>
#include <serializer.h>
#include <memory.h>
#include <marching_squares.h>
#include <graphy.h>
#define RUN_TESTS

#if defined(RUN_TESTS)
#include <tests.h>

void run_self_tests(){
    test_uniform_grid2D();
    test_uniform_grid3D();
    test_distribute_uniform_grid2D();
    test_distribute_uniform_grid3D();
    test_distribute_random_grid2D();
    test_distribute_random_grid3D();
    test_neighbor_query_grid2D();
    test_neighbor_query_grid3D();
    test_kernels_2D();
    test_kernels_3D();
    test_triangle_point_generator();
    //test_matrix_operations(); // precision is killing this test for 32 bits
    test_ray2_intersect();
    test_closest_point_sphere2D();
    test_color_field_2D();
    test_rectangle_distance_inside();
    test_rectangle_distance_outside();
    test_rectangle_emit();
    test_field_grid();
    test_grid_face_centered_2D();
    test_mesh_collision();
    test_box_distance();
    test_espic_1D();
    test_node_grid_2D();
    test_particle_to_node_2D();
    test_espic_particles_2D();
    test_sph2_water_block();
    test_sph2_water_sphere();
    test_sph2_double_dam_break();
    test_sph2_water_drop();
    test_sph2_gas_sphere();
    test_sph3_water_sphere();
    test_sph3_water_block();
    test_sph3_double_dam_break();
    
    test_pcisph2_double_dam_break();
    test_pcisph2_water_sphere();
    test_pcisph2_water_drop();
    test_pcisph2_water_block();
    
    test_bcclattice_point_generator();
    test_pcisph3_water_sphere();
    test_pcisph3_double_dam_break();
    test_pcisph3_whale_obstacle();
    test_pcisph3_dragon();
    test_pcisph3_quadruple_dam();
    test_pcisph3_ball_many_emission();
    test_pcisph3_dragon_shower();
}
#endif
void test_pcisph2_marching_squares();
void test(){
    FieldGrid2f grid;
    grid.Build(vec2ui(4, 4), vec2f(0.1), vec2f(-0.2, -0.2), VertexCentered);
    
    vec2ui res = grid.resolution;
    for(int i = 0; i < res.x; i++){
        for(int j = 0; j < res.y; j++){
            if(i > 0 && j > 0 && i < res.x - 1 && j < res.y - 1){
                grid.SetValueAt(-2, vec2ui(i, j));
            }else{
                grid.SetValueAt(2, vec2ui(i, j));
            }
        }
    }
    
    std::vector<vec3f> triangles;
    MarchingSquares(&grid, 0, &triangles);
    int totalLines = triangles.size() * 3;
    float *pos = new float[totalLines * 3];
    
    int it = 0;
    for(int i = 0; i < triangles.size()/3; i++){
        vec3f p0 = triangles[3 * i + 0];
        vec3f p1 = triangles[3 * i + 1];
        vec3f p2 = triangles[3 * i + 2];
        
        pos[it++] = p0.x; pos[it++] = p0.y; pos[it++] = p0.z;
        pos[it++] = p1.x; pos[it++] = p1.y; pos[it++] = p1.z;
        pos[it++] = p1.x; pos[it++] = p1.y; pos[it++] = p1.z;
        pos[it++] = p2.x; pos[it++] = p2.y; pos[it++] = p2.z;
        pos[it++] = p2.x; pos[it++] = p2.y; pos[it++] = p2.z;
        pos[it++] = p0.x; pos[it++] = p0.y; pos[it++] = p0.z;
    }
    
    float rgb[3] = {1, 0, 0};
    graphy_set_orthographic(-0.2, 0.2, -0.2, 0.2);
    graphy_render_lines(pos, rgb, totalLines);
    getchar();
    graphy_close_display();
    delete[] pos;
}

void test_lnm_happy_whale();
void test_pcisph3_dragon_pool();
int main(int argc, char **argv){
    printf("* Bubbles Fluid Simulator - Built %s at %s *\n", __DATE__, __TIME__);
    cudaInitEx();
    
    //test();
    //test_pcisph2_marching_squares();
    //test_pcisph3_box_many_emission();
	//test_pcisph3_quadruple_dam();
    test_pcisph3_double_dam_break();
    //test_pcisph3_water_drop();
    //test_pcisph3_happy_whale();
    //test_lnm_happy_whale();
    //test_pcisph2_water_block_lnm();
    //test_pcisph3_dragon_pool();
    //test_pcisph3_dragon_shower();
#if defined(RUN_TESTS)
    //run_self_tests();
#else
    
#endif
    CudaMemoryManagerClearAll();
    graphy_close_display();
    cudaSafeExit();
    return 0;
}
