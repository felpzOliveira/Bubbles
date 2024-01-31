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


void test_pcisph3_box_drop(){
    printf("===== PCISPH Solver 3D -- Box Drop\n");
    CudaMemoryManagerStart(__FUNCTION__);

    vec3f origin(4.0, 0, 0);
    vec3f target(0.0f);
    vec3f containerSize(2.0);
    vec3f boxEmitSize(0.4, 0.1, 0.4);
    vec3f boxSize0(0.4);
    Float spacing = 0.02;
    Float spacingScale = 1.8;

    Shape *container = MakeBox(Transform(), containerSize, true);

    Float y0of = (containerSize.y - boxSize0.y) * 0.5; y0of -= spacing;
    Shape *box0 = MakeBox(RotateY(45) * RotateX(45.0), boxSize0);

    Float yEof = (containerSize.y - boxEmitSize.y) * 0.5; yEof -= spacing;
    Shape *boxEmitter = MakeBox(Translate(0, yEof, 0), boxEmitSize);

    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(box0);
    cBuilder.AddCollider3(container);

    ContinuousParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    emitterSet.AddEmitter(boxEmitter, spacing);

    pBuilder.SetKernelRadius(spacing * spacingScale);
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    Assure(UtilIsEmitterOverlapping(&emitterSet, colliders) == 0);

    emitterSet.Emit(&pBuilder);
    SphParticleSet3 *sphSet = SphParticleSet3FromContinuousBuilder(&pBuilder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    Grid3 *domainGrid = UtilBuildGridForDomain(container->GetBounds(),
                                               spacing, spacingScale);

    PciSphSolver3 solver;
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, domainGrid, sphSet);
    solver.SetColliders(colliders);

    Float targetInterval =  1.0 / 240.0;
    pBuilder.MapGrid(domainGrid);
    int extraParts = 24 * 10;

    auto velocityField = [&](const vec3f &p) -> vec3f{
        Float u1 = rand_float();
        Float u2 = rand_float();
        Float sign1 = rand_float() < 0.5 ? 1 : -1;
        Float sign2 = rand_float() < 0.5 ? 1 : -1;
        return vec3f(u1 * sign1 * 5.f, -10, u2 * sign2 * 5.f);
    };

    auto filler = [&](float *pos, float *col) -> int{
        int f = UtilGenerateBoxPoints(pos, col, vec3f(1,1,0), containerSize,
                                      extraParts * 0.8, container->ObjectToWorld);
        f += UtilGenerateBoxPoints(&pos[3 * f], &col[3 * f], vec3f(1,1,0), boxSize0,
                                   extraParts * 0.2, box0->ObjectToWorld);
        return f;
    };

    auto colorFunction = [&](float *colors, int pCount) -> void{
        for(int i = 0; i < pCount; i++){
            colors[3 * i + 0] = 1; colors[3 * i + 1] = 0;
            colors[3 * i + 2] = 1;
        }
    };

    auto onStepUpdate = [&](int step) -> int{
        if(step == 0) return 1;
        if(pSet->GetParticleCount() < 300000){
            pBuilder.MapGridEmit(velocityField, spacing);
        }

        UtilPrintStepSimple(&solver, step-1);
#if 0
        std::string path("/media/felipe/FluidStuff/box/out_");
        path += std::to_string(step-1);
        path += ".txt";
        SerializerSaveSphDataSet3Legacy(solver.GetSphSolverData(), path.c_str(),
                                        SERIALIZER_POSITION);
#endif
        return 1;
    };

    UtilRunDynamicSimulation3<PciSphSolver3, ParticleSet3>(&solver, pSet, spacing, origin,
                                                           target, targetInterval, extraParts,
                                                           {}, onStepUpdate, colorFunction,
                                                           filler);

    CudaMemoryManagerClearCurrent();

    printf("===== OK\n");
}

