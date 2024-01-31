#include <pcisph_solver.h>
#include <emitter.h>
#include <tests.h>
#include <grid.h>
#include <graphy.h>
#include <serializer.h>
#include <string>
#include <unistd.h>
#include <obj_loader.h>
#include <util.h>
#include <memory.h>
#include <transform_sequence.h>
#include <sdfs.h>
#include <dilts.h>
#include <marching_cubes.h>

void set_particle_color(float *pos, float *col, ParticleSet3 *pSet);
void simple_color(float *pos, float *col, ParticleSet3 *pSet);

void test_pcisph3_progressive(){
    printf("===== PCISPH Solver 3D -- Progressive\n");
    CudaMemoryManagerStart(__FUNCTION__);

    const char *pFile = "output.txt";
    const int maxParticles = 5.0 * kDefaultMaxParticles;
    ProgressiveParticleSetBuilder3 tmpBuilder(maxParticles), pBuilder(maxParticles);

    int flags = SERIALIZER_POSITION;
    std::vector<vec3f> points;
    SerializerLoadPoints3(&points, pFile, flags);

    ColliderSetBuilder3 cBuilder;
    vec3f origin(-3.0f, 0.f, 4.f);
    vec3f target(0.0f);
    vec3f containerSize;

    Float spacing = 0.012;
    Float spacingScale = 1.8;

    Bounds3f bounds;

    bounds = Bounds3f(points[0]);
    for(vec3f p : points){
        bounds = Union(bounds, p);
    }

    vec3f offset = bounds.Center();
    bounds = Bounds3f(points[0] - offset);

    tmpBuilder.SetKernelRadius(spacing * spacingScale);
    pBuilder.SetKernelRadius(spacing * spacingScale);

    for(int i = 1; i < points.size(); i++){
        vec3f pi = points[i] - offset;
        tmpBuilder.AddParticle(pi);
        bounds = Union(bounds, pi);
    }

    tmpBuilder.Commit();
    bounds.Expand(5.0f * spacing * spacingScale);

    containerSize = bounds.Size();

    Shape *container = MakeBox(Transform(), containerSize, true);

    //const char *meshPath = "Meshes/test_mesh_1.obj";
    //ParsedMesh *pmesh = LoadObj(meshPath);
    //Shape *meshShape = MakeMesh(pmesh, Translate(-offset.x, -offset.y, -offset.z));
    //GenerateShapeSDF(meshShape, 0.03, 0.03);

    //cBuilder.AddCollider3(meshShape);
    cBuilder.AddCollider3(container);

    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    SphParticleSet3 *tmpSphSet = SphParticleSet3FromProgressiveBuilder(&tmpBuilder);
    SphParticleSet3 *sphSet  = SphParticleSet3FromProgressiveBuilder(&pBuilder);

    Grid3 *mapGrid = UtilBuildGridForDomain(container->GetBounds(),
                                            spacing, spacingScale);
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(),
                                               spacing, spacingScale);

    ResetAndDistribute(mapGrid, tmpSphSet->GetParticleSet());
    ResetAndDistribute(domainGrid, sphSet->GetParticleSet());

    tmpBuilder.MapGrid(mapGrid);
    pBuilder.MapGrid(domainGrid);

    Bounds3f boundList[] = {Bounds3f(vec3f(-1.95255, 0.700118, -1.9521) - offset,
                                  vec3f(2.04845, 0.900118, 0.471892) - offset) };

    tmpBuilder.MapGridEmitToOther(&pBuilder, ZeroVelocityField3, boundList[0]);

    ParticleSet3 *pSet = sphSet->GetParticleSet();
    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    pSet->SetUserVec3Buffer(1);
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec3f p = pSet->GetParticlePosition(i);
        if(p.x > 0)
            pSet->SetParticleUserBufferVec3(i, vec3f(1,0,0), 0);
        else
            pSet->SetParticleUserBufferVec3(i, vec3f(0,1,0), 0);
    }

    Assure(UtilIsDistributionConsistent(pSet, domainGrid) == 1);
    Float targetInterval =  1.0 / 240.0;

    int count = pSet->GetParticleCount();
    AssureA(count > 0, "No initial particles in the system!");

    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    memset(col, 0, sizeof(float) * 3 * count);

    graphy_vector_set(origin, target);
    simple_color(pos, col, pSet);

    graphy_render_points3f(pos, col, count, spacing/2.0);

    int frame_index = 0;
    while(true){
        solver.Advance(targetInterval);
        simple_color(pos, col, pSet);
        graphy_render_points3f(pos, col, count, spacing/2.0);
        frame_index++;
        if(frame_index == 100 && false){
            tmpBuilder.MapGridEmitToOther(&pBuilder, ZeroVelocityField3, boundList[1]);
            count = pSet->GetParticleCount();
            delete[] col;
            delete[] pos;
            pos = new float[count * 3];
            col = new float[count * 3];
            memset(col, 0, sizeof(float) * 3 * count);
        }

        printf("Current frame= %d\n", frame_index);
    }

    CudaMemoryManagerClearCurrent();
    printf("===== OK\n");
}
