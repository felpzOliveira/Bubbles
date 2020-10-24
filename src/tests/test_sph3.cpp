#include <sph_solver.h>
#include <graphy.h>
#include <transform.h>
#include <emitter.h>
#include <tests.h>
#include <unistd.h>

void graphy_vector_set(vec3f origin, vec3f target, Float fov, Float near, Float far){
    graphy_set_3d(origin.x, origin.y, origin.z, target.x, target.y, target.z,
                  fov, near, far);
}

void graphy_vector_set(vec3f origin, vec3f target){
    graphy_set_3d(origin.x, origin.y, origin.z, target.x, target.y, target.z,
                  45.0, 0.1f, 100.0f);
}

void simple_color(float *pos, float *col, ParticleSet3 *pSet){
    int count = pSet->GetParticleCount();
    for(int i = 0; i < count; i++){
        vec3f pi = pSet->GetParticlePosition(i);
        pos[3 * i + 0] = pi.x; pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = pi.z; col[3 * i + 0] = 1;
    }
}

void test_sph3_double_dam_break(){
    printf("===== SPH Solver 3D -- Double Dam Break\n");
    vec3f origin(1, 2, 0);
    vec3f target(0,0,0);
    
    Float spacing = 0.02;
    Float boxLen = 1.5;
    Float boxFluidLen = 0.5;
    Float boxFluidYLen = 0.9;
    Float spacingScale = 1.8;
    
    /* Build shapes */
    Float xof = (boxLen - boxFluidLen)/2.0; xof -= spacing;
    Float zof = (boxLen - boxFluidLen)/2.0; zof -= spacing;
    Float yof = (boxLen - boxFluidYLen)/2.0; yof -= spacing;
    
    vec3f boxSize = vec3f(boxFluidLen, boxFluidYLen, boxFluidLen);
    
    Shape *container = MakeBox(Transform(), vec3f(boxLen), true);
    Shape *boxp = MakeBox(Translate(xof, -yof, zof), boxSize);
    Shape *boxn = MakeBox(Translate(-xof, -yof, -zof), boxSize);
    
    /* Emit particles */
    ParticleSetBuilder3 pBuilder;
    VolumeParticleEmitterSet3 emitterSet;
    
    VolumeParticleEmitter3 emitterp(boxp, boxp->GetBounds(), spacing);
    VolumeParticleEmitter3 emittern(boxn, boxn->GetBounds(), spacing);
    
    emitterSet.AddEmitter(&emitterp);
    emitterSet.AddEmitter(&emittern);
    
    emitterSet.Emit(&pBuilder);
    
    /* Build domain and colliders */
    Bounds3f containerBounds = container->GetBounds();
    vec3f pMin = containerBounds.pMin - vec3f(spacing);
    vec3f pMax = containerBounds.pMax + vec3f(spacing);
    
    int resolution = (int)std::floor(boxLen / (spacing * spacingScale));
    Grid3 *grid = MakeGrid(vec3ui(resolution), pMin, pMax);
    
    ColliderSetBuilder3 cBuilder;
    cBuilder.AddCollider3(container);
    
    ColliderSet3 *colliders = cBuilder.GetColliderSet();
    
    /* Setup solver */
    SphSolver3 solver;
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&pBuilder);
    
    solver.Initialize(DefaultSphSolverData3());
    solver.Setup(WaterDensity, spacing, spacingScale, grid, sphSet);
    solver.SetColliders(colliders);
    
    /* Set timestep and view stuff */
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0, sizeof(float) * 3 * count);
    SphSolverData3 *data = solver.GetSphSolverData();
    graphy_vector_set(origin, target);
    
    simple_color(pos, col, pSet);
    graphy_render_points3f(pos, col, count, 0.01);
    
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver.Advance(targetInterval);
        simple_color(pos, col, pSet);
        graphy_render_points3f(pos, col, count, 0.01);
        printf("Step: %d\n", i+1);
    }
    
    
    delete[] pos;
    delete[] col;
    emitterSet.Release();
    pBuilder.Release();
    cudaFree(container);
    cudaFree(boxp);
    cudaFree(boxn);
    printf("===== OK\n");
}

void test_sph3_water_block(){
    printf("===== SPH Solver 3D -- Water Block\n");
    vec3f origin(1,2,0);
    vec3f target(0,0,0);
    Float spacing = 0.02;
    Float fHeight = 0.5;
    Float bHeight = 2.0;
    
    Float yOffset = (bHeight - fHeight) / 2.0;
    yOffset -= spacing;
    
    Shape *box = MakeBox(Translate(0, -yOffset, 0), vec3f(1,fHeight,1));
    Shape *container = MakeBox(Translate(0), vec3f(bHeight), true);
    //Shape *container = MakeSphere(Transform(), bHeight/2.0f, true);
    ParticleSetBuilder3 builder;
    ColliderSetBuilder3 colliderBuilder;
    
    SphSolver3 solver;
    int reso = (int)std::floor(bHeight / (spacing * 1.8));
    printf("Using grid with resolution %d x %d x %d\n", reso, reso, reso);
    vec3ui res(reso);
    
    solver.Initialize(DefaultSphSolverData3());
    
    Bounds3f containerBounds = container->GetBounds();
    vec3f pMin = containerBounds.pMin - vec3f(spacing);
    vec3f pMax = containerBounds.pMax + vec3f(spacing);
    
    Grid3 *grid = MakeGrid(res, pMin, pMax);
    VolumeParticleEmitter3 emitter(box, box->GetBounds(), spacing);
    emitter.Emit(&builder);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    
    int count = pSet->GetParticleCount();
    
    colliderBuilder.AddCollider3(container);
    ColliderSet3 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(WaterDensity, spacing, 1.8, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0, sizeof(float) * 3 * count);
    SphSolverData3 *data = solver.GetSphSolverData();
    graphy_vector_set(origin, target);
    
    simple_color(pos, col, pSet);
    graphy_render_points3f(pos, col, count, 0.01);
    
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver.Advance(targetInterval);
        simple_color(pos, col, pSet);
        graphy_render_points3f(pos, col, count, 0.01);
        printf("Step: %d\n", i+1);
    }
    
    printf("===== OK\n");
}

void test_sph3_water_sphere(){
    printf("===== SPH Solver 3D -- Water in Ball\n");
    vec3f origin(0,0,3);
    vec3f target(0,0,0);
    
    Float spacing = 0.02;
    //Float spacing = 0.05;
    Shape *sphere = MakeSphere(Translate(-0.4,0.25,0.0), 0.5);
    Shape *container = MakeSphere(Transform(), 1.0, true);
    
    ParticleSetBuilder3 builder;
    ColliderSetBuilder3 colliderBuilder;
    
    SphSolver3 solver;
    int reso = (int)std::floor(2 * 1 / (spacing * 1.8));
    printf("Using grid with resolution %d x %d x %d\n", reso, reso, reso);
    vec3ui res(reso);
    
    solver.Initialize(DefaultSphSolverData3());
    
    Bounds3f containerBounds = container->GetBounds();
    vec3f pMin = containerBounds.pMin - vec3f(spacing);
    vec3f pMax = containerBounds.pMax + vec3f(spacing);
    
    Grid3 *grid = MakeGrid(res, pMin, pMax);
    
    VolumeParticleEmitter3 emitter(sphere, sphere->GetBounds(), spacing);
    emitter.Emit(&builder);
    
    SphParticleSet3 *sphSet = SphParticleSet3FromBuilder(&builder);
    ParticleSet3 *pSet = sphSet->GetParticleSet();
    
    int count = pSet->GetParticleCount();
    
    colliderBuilder.AddCollider3(container);
    ColliderSet3 *collider = colliderBuilder.GetColliderSet();
    
    solver.Setup(WaterDensity, spacing, 1.8, grid, sphSet);
    solver.SetColliders(collider);
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0, sizeof(float) * 3 * count);
    SphSolverData3 *data = solver.GetSphSolverData();
    
    graphy_vector_set(origin, target);
    
    simple_color(pos, col, pSet);
    graphy_render_points3f(pos, col, count, 0.01);
    
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver.Advance(targetInterval);
        simple_color(pos, col, pSet);
        
        graphy_render_points3f(pos, col, count, 0.01);
        printf("Step: %d\n", i+1);
    }
    
    printf("===== OK\n");
}