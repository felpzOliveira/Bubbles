#include <pbf_solver.h>
#include <cutil.h>
#include <particle.h>
#include <grid.h>
#include <emitter.h>
#include <graphy.h>
#include <tests.h>


void test_pbf2_double_dam_break(){
    printf("===== PBF Solver 2D -- Double Dam Break\n");
    Float spacing = 0.02;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    Grid2 *grid = cudaAllocateVx(Grid2, 1);
    PbfSolver2 *solver = cudaAllocateVx(PbfSolver2, 1);
    
    int reso = (int)std::floor(lenc / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    Float boxLenx = 0.4;
    Float boxLeny = 1.0;
    vec2f boxDim(boxLenx, boxLeny);
    
    solver->Initialize(DefaultSphSolverData2());
    Shape2 *rect = MakeRectangle2(Translate2(-(lenc - boxLenx)/2.f + spacing, 
                                             -(boxLeny/2.f - spacing)), boxDim);
    
    Shape2 *rect2 = MakeRectangle2(Translate2((lenc - boxLenx)/2.f - spacing, 
                                              -(boxLeny/2.f-spacing)), boxDim);
    
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    grid->Build(res, pMin, pMax);
    
    VolumeParticleEmitterSet2 emitterSet;
    
    emitterSet.AddEmitter(rect, rect->GetBounds(), spacing);
    emitterSet.AddEmitter(rect2, rect2->GetBounds(), spacing);
    
    emitterSet.Emit(&builder);
    
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver->Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver->SetColliders(collider);
    
    Float targetInterval = 1.0 / 800.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0x00, sizeof(float) * 3 * count);
    SphSolverData2 *data = solver->GetSphSolverData();
    set_colors_lnm(col, data);
    
    for(int i = 0; i < 20 * 26 * 100; i++){
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
        solver->Advance(targetInterval);
        ProfilerReport();
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}
