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
    test_pcisph3_quadruple_dam();
    test_pcisph3_happy_whale();
    test_pcisph3_dragon_pool();
    test_pcisph3_sdf();
}
#endif

void test_pcisph3_pathing();
void test_pcisph3_box_drop();
void test_bounds_split3();
void test_pcisph3_water_block();
void test_pcisph3_dissolve();
void test_pcisph3_emit_test();
void test_pcisph2_progressive_emitter();

bb_cpu_gpu Float BVHMeshBoundedClosestDistance(const vec3f &point, int *closest,
                                         ParsedMesh *mesh, Node *bvh, Bounds3f bounds);

void load_test(std::vector<Shape *> &vShapes, std::vector<Bounds3f> &vBounds){
    const char *meshPath = "Meshes/test_mesh_";
    for(int i = 1; i < 30; i++){
        Float targetScale = 0;
        std::string mesh_i = std::string(meshPath);
        mesh_i += std::to_string(i);
        mesh_i += ".obj";
        ParsedMesh *pmesh = LoadObj(mesh_i.c_str());
        Transform scale = UtilComputeFitTransform(pmesh, 1.9f, &targetScale);

        Shape *shape = MakeMesh(pmesh, scale);
        GenerateShapeSDF(shape, 0.01, 0.01);

        vShapes.push_back(shape);
        printf(" >> Loaded %d\n", i);

        std::string boundsPath(meshPath);
        boundsPath += std::to_string(i) + std::string("_bounds.txt");
        FILE *fp = fopen(boundsPath.c_str(), "rb");
        if(!fp){
            printf("Could not load bounds\n");
            exit(0);
        }

        Bounds3f bounds;
        fscanf(fp, "%g %g %g\n", &bounds.pMin.x, &bounds.pMin.y, &bounds.pMin.z);
        fscanf(fp, "%g %g %g\n", &bounds.pMax.x, &bounds.pMax.y, &bounds.pMax.z);

        vBounds.push_back(bounds);

        fclose(fp);
        bounds.PrintSelf();
        std::cout << std::endl;
    }
}

#include <marching_cubes.h>
void test_split(){
    CudaMemoryManagerStart(__FUNCTION__);

    Float targetScale = 0;
    const char *meshPath = "/home/felpz/Documents/CGStuff/models/head.obj";
    ParsedMesh *pmesh = LoadObj(meshPath);
    Transform scale = UtilComputeFitTransform(pmesh, 4.0f, &targetScale);
    printf("Target scale = %g\n", targetScale);

    Shape *shape = MakeMesh(pmesh, scale);
    GenerateShapeSDF(shape, 0.005, 0.01);

    printf("Updating...\n");
    FieldGrid3f *grid = nullptr;
    Node *bvh = shape->bvh;
    Float height = bvh->bound.ExtentOn(1);
    int stepCount = 30;
    Float dy = height / (Float)stepCount;

    HostTriangleMesh3 mesh0;
    printf("Running Marching Cubes { 0 }\n");

    MarchingCubes(shape->grid, &mesh0, 0.f, false);
    mesh0.writeToDisk("Meshes/test_mesh_0.obj", FORMAT_OBJ);

    for(int i = 1; i <= stepCount; i++){
        vec3f pMax = shape->GetBounds().pMax;
        vec3f pMin = shape->GetBounds().pMin;
        Bounds3f cbounds(vec3f(pMin.x-0.001, pMax.y - i * dy, pMin.z-0.001), pMax);

        grid = UpdatedSDFToOther(shape->grid, grid,
        GPU_LAMBDA(vec3f point, FieldGrid3f *field, int index) -> Float{
            if(index == 0){
                printf("{%g %g %g} x {%g %g %g}\n",
                       cbounds.pMin.x, cbounds.pMin.y, cbounds.pMin.z,
                       cbounds.pMax.x, cbounds.pMax.y, cbounds.pMax.z);
            }
            ParsedMesh *mesh = shape->mesh;
            Node *bvh = shape->bvh;
            Float currSdf = field->Sample(point);
            int closestId = -1;
            Float sign = 1.f;
            if(!InsideExclusive(point, cbounds) && currSdf < 0)
                sign = -1.f;

            Float dist =
                BVHMeshBoundedClosestDistance(point, &closestId, mesh, bvh, cbounds);
            Float psd = Max(Absf(dist), 0.00001);
            return sign * psd;
        });

        printf("Running Marching Cubes { %d }\n", i);
        HostTriangleMesh3 meshi;
        MarchingCubes(grid, &meshi, 0.f, false);
        std::string path("Meshes/test_mesh_");
        path += std::to_string(i);
        std::string meshPath = path + ".obj";

        std::string boundsPath = path + "_bounds.txt";

        meshi.writeToDisk(meshPath.c_str(), FORMAT_OBJ);

        std::ofstream ofs(boundsPath);
        if(ofs.is_open()){
            ofs << cbounds.pMin.x << " " << cbounds.pMin.y << " " << cbounds.pMin.z << std::endl;
            ofs << cbounds.pMax.x << " " << cbounds.pMax.y << " " << cbounds.pMax.z << std::endl;
            ofs.close();
        }
    }

    printf("OK\n");

    CudaMemoryManagerClearCurrent();
}


void test_pcisph3_prog();
int main(int argc, char **argv){
    BB_MSG("Bubbles Fluid Simulator");
    /* Initialize cuda API */
    cudaInitEx();

    /* Sets the default kernel launching parameters */
    cudaSetLaunchStrategy(CudaLaunchStrategy::CustomizedBlockSize, 16);

    //std::vector<Shape *> shapes;
    //std::vector<Bounds3f> bounds;
    //test_split();
    //load_test(shapes, bounds);

    //test_pcisph3_box_drop();
    //test_pcisph3_prog();
    test_pcisph3_progressive();
    //test_pcisph2_progressive_emitter();
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
