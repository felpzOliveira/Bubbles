#include <cutil.h>
#include <transform.h>
#include <particle.h>
#include <emitter.h>
#include <obj_loader.h>
#include <serializer.h>
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
    test_pcisph2_double_dam_break();
    test_pcisph2_water_sphere();
    test_pcisph2_water_drop();
    test_pcisph2_water_block();
    
    test_bcclattice_point_generator();
    test_pcisph3_water_sphere();
}
#endif

void MeshToParticles(const char *name, const Transform &transform,
                     Float spacing, const char *output)
{
    printf("===== Emitting particles from mesh\n");
    
    UseDefaultAllocatorFor(AllocatorType::CPU);
    ParsedMesh *mesh = LoadObj(name);
    
    Shape *shape = MakeMesh(mesh, transform);
    ParticleSetBuilder3 builder;
    VolumeParticleEmitter3 emitter(shape, shape->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    SphSolverData3 *data = DefaultSphSolverData3();
    data->sphpSet = sphSet;
    data->domain = nullptr;
    
    SerializerSaveSphDataSet3(data, output, SERIALIZER_POSITION);
    
    printf("===== OK\n");
    exit(0);
}

int main(int argc, char **argv){
    cudaInitEx();
#if 0
    MeshToParticles("/home/felpz/Documents/models/dragon_aligned.obj",
                    Scale(0.7) * RotateY(-90), 0.02, "output.txt");
    MeshToParticles("/home/felpz/Downloads/happy_whale.obj",
                    Scale(0.3), 0.02, "output.txt");
#endif
    
    //test_pcisph3_water_sphere();
    //test_pcisph3_dragon();
    test_pcisph3_happy_whale();
    //test_display_set();
    
#if defined(RUN_TESTS)
    //run_self_tests();
#else
    
#endif
    cudaSafeExit();
    return 0;
}