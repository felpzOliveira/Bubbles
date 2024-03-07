#include <cutil.h>
#include <transform.h>
#include <particle.h>
#include <emitter.h>
#include <obj_loader.h>
#include <serializer.h>
#include <memory.h>
#include <marching_squares.h>
#include <graphy.h>
#include <util.h>

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
    test_bounds_split2();
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
    test_pcisph2_water_block_lnm();
    test_pcisph2_continuous_emitter();

    test_bcclattice_point_generator();
    test_pcisph3_water_sphere();
    test_pcisph3_double_dam_break();
    test_pcisph3_whale_obstacle();
    test_pcisph3_water_drop();
    test_pcisph3_dragon();
    test_pcisph3_quadruple_dam();
    test_pcisph3_ball_many_emission();
    test_pcisph3_dragon_shower();
    test_pcisph3_box_many_emission();
    test_pcisph3_happy_whale();
    test_pcisph3_dragon_pool();
    test_pcisph3_sdf();
}
#endif

void test_routine(){
    Bounds3f bounds;
    std::string basePath("../simulations/gravity/out_");
    int total = 1432;
    std::cout << bounds << std::endl;
    for(int i = 0; i < total; i++){
        std::vector<vec3f> points;
        int flags = SERIALIZER_POSITION;
        std::string filename = basePath + std::to_string(i);
        filename += ".txt";
        SerializerLoadLegacySystem3(&points, filename.c_str(), flags);

        for(vec3f p : points){
            bounds = Union(bounds, p);
        }

        std::cout << (i+1) << "/" << total << " {" << points.size() << "}" << std::endl;
    }

    std::cout << bounds << std::endl;
}

int main(int argc, char **argv){
    BB_MSG("Bubbles Fluid Simulator");
    /* Initialize cuda API */
    cudaInitEx();

    /* Sets the default kernel launching parameters */
    cudaSetLaunchStrategy(CudaLaunchStrategy::CustomizedBlockSize, 16);

    /* Path to where to find models */
    UtilSetGlobalModelPath("/home/felpz/Documents/CGStuff/models");

    /* Path to where to write file output */
    UtilSetGlobalOutputPath("../simulations");

    /* Disable file output */
    SerializerSetWrites(false);

    for(int i = 1; i < argc; i++){
        std::string arg(argv[i]);
        if(arg == "--enable-output")
            SerializerSetWrites(true);
    }

    if(!SerializerIsWrittable()){
        printf("(Note): Serializer is not writtable, will not output.\n"
               "        Consider using --enable-output.\n");
    }

    //test_routine();
    //test_pcisph2_helix();
    //test_pcisph3_helix();
    //test_pcisph3_dam_break();
    //test_pcisph3_box_drop();
    //test_pcisph3_gravity_field();
    //test_pcisph2_water_block();
    //test_pcisph2_marching_squares();

    //test_pcisph2_continuous_emitter();
    //test_pcisph2_double_dam_break();
    //test_pcisph2_water_block_lnm();
    //test_pcisph3_sdf();
    //test_pcisph3_rotating_water_box();
    //test_pcisph3_dam_break_double_dragon();
    //test_pcisph3_quadruple_dam();
    //test_pcisph3_pathing();

    //test_pcisph3_box_drop();
    //test_pcisph3_happy_whale();
    //test_pcisph3_water_drop();

    //test_pcisph2_water_block();
    //test_pcisph3_dragon();
    //test_pcisph2_water_sphere();
    //test_pcisph3_water_block();
    //test_pcisph3_dam_break_double_dragon();

    //test_pcisph3_box_mesh();
    //test_pcisph3_dragon_pool();
    //test_pcisph3_double_dam_break();
    //test_pcisph3_rotating_water_box();
    //test_pcisph3_dragon_shower();

    //test_pcisph3_dissolve();
    //test_pcisph3_emit_test();
    //test_pcisph3_water_drop();
    //test_pcisph2_water_sphere();
    //test_pcisph2_marching_squares();

#if defined(RUN_TESTS)
    //run_self_tests();
#else

#endif
    CudaMemoryManagerClearAll();
    graphy_close_display();
    cudaSafeExit();
    return 0;
}
