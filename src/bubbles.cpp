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
#include <marching_cubes.h>
#include <sdfs.h>

#define RUN_TESTS

extern std::string GraphyPath;

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

// NOTE: can use this routines for generating the teddies/origami scene.
void sdf_teddies(){
    HostTriangleMesh3 mesh;
    Float iso = 0.0;
    Float dx  = 0.02;
    Bounds3f bounds(vec3f(-5), vec3f(5));
    FieldGrid3f *field = CreateSDF(bounds, dx, AutoLambda(vec3f point){
        //return T_OrigamiBoat(point, -1);
        //return T_OrigamiDragon(point);
        //return T_OrigamiWhale(point, 2);
        //return Teddy_Lying(point);
        return Teddy_Sitting(point);
        //return Teddy_Standing(point);
    });

    vec3ui res = field->GetResolution();
    printf("Resolution= {%u %u %u}\n", res.x, res.y, res.z);
#if 0
    Bounds3f reducedB(vec3f(-1), vec3f(1));

    Transform transform = Scale(vec3f(5.f));
    auto sample_fn = GPU_LAMBDA(vec3f point, Shape *, int) -> Float{
        vec3f query = transform.Point(point);
        return field->Sample(query);
    };

    Shape *testShape = MakeSDFShape(reducedB, sample_fn);

    MarchingCubes(testShape->grid, &mesh, iso, false);
#else
    MarchingCubes(field, &mesh, iso, false);
#endif
    mesh.writeToDisk("test_sdf.obj", FORMAT_PLY);
    exit(0);
}

void test_pcisph3_water_drop2();
void test_pcisph3_gravity_field2();
void test_pcisph3_tank_dam_paper();

int main(int argc, char **argv){
    BB_MSG("Bubbles Fluid Simulator");
    /* Initialize cuda API */
    cudaInitEx();

    /* Sets the default kernel launching parameters */
    cudaSetLaunchStrategy(CudaLaunchStrategy::CustomizedBlockSize, 16);

    /* Disable file output */
    SerializerSetWrites(false);

    std::string modelsPath;
    std::string outputPath("../simulations");

    for(int i = 1; i < argc; i++){
        std::string arg(argv[i]);
        if(arg == "--enable-output")
            SerializerSetWrites(true);
        else if(arg.substr(0, 9) == "--graphy="){
            GraphyPath = arg.substr(9);
        }else if(arg.substr(0, 9) == "--models="){
            modelsPath = arg.substr(9);
        }else if(arg.substr(0, 10) == "--outpath="){
            outputPath = arg.substr(9);
        }
    }

    /* Path to where to find models */
    if(modelsPath.size() > 0)
        UtilSetGlobalModelPath(modelsPath.c_str());

    /* Path to where to write file output */
    UtilSetGlobalOutputPath(outputPath.c_str());


    if(!SerializerIsWrittable()){
        printf("(Note): Serializer is not writtable, will not output.\n"
               "        Consider using --enable-output.\n");
    }
    //sdf_teddies();

    //test_svd();
    //test_pcisph2_water_sphere_dynamic();

    //test_sdf_teddies();
    //test_routine(1400);
    //test_pcisph2_helix();
    //test_pcisph3_helix();
    //test_pcisph3_dam_break();
    //test_pcisph3_box_drop();
    //test_pcisph3_gravity_field();
    //test_pcisph2_water_block();
    //test_pcisph2_marching_squares();
    //test_pbf2_double_dam_break();
    //test_pcisph3_water_sphere();
    //test_lnm_happy_whale();
    //test_pcisph3_water_sphere_movable();

    //test_pcisph2_continuous_emitter();
    //test_pcisph2_double_dam_break();
    //test_pcisph2_water_drop();
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
    //test_pcisph2_water_sphere_dynamic();
    //test_pcisph3_water_block();
    //test_pcisph3_dam_break_double_dragon();
    //test_pcisph3_water_sphere_movable();
    test_pcisph3_tank_dam_paper();

    //test_pcisph3_dragon_pool();
    //test_pcisph3_tank();
    //test_pcisph3_tank_dam();
    //test_pcisph3_double_dam_break();
    //test_pcisph3_rotating_water_box();
    //test_pcisph3_dragon_shower();
    //test_pcisph3_water_drop2();

    //test_pcisph3_dissolve();
    //test_pcisph3_emit_test();
    //test_pcisph3_water_drop();
    //test_pcisph2_water_sphere();
    //test_pcisph2_marching_squares();
    //test_pcisph3_ball_many_emission();
    //test_pcisph3_water_sphere_movable();
    //test_pcisph3_gravity_field2();

#if defined(RUN_TESTS)
    //run_self_tests();
#else

#endif
    CudaMemoryManagerClearAll();
    graphy_close_display();
    cudaSafeExit();
    return 0;
}
