#include <cutil.h>
#include <particle.h>
#include <grid.h>
#include <sph_solver.h>
#include <emitter.h>
#include <graphy.h>
#include <tests.h>

vec3f get_color_by_hex(unsigned int hex){
    unsigned int r = (hex & 0x00ff0000) >> 16;
    unsigned int g = (hex & 0x0000ff00) >> 8;
    unsigned int b = (hex & 0x000000ff);
    Float inv = 1.f / 255.f;
    return vec3f(r * inv, g * inv, b * inv);
}

vec3f get_color_level(int level){
    vec3f color_map[] = {
        get_color_by_hex(0x08519c), get_color_by_hex(0x3182bd),
        get_color_by_hex(0x6baed6),
    };
    
    level -= 1;
    if(level < 2 && level >= 0) return color_map[level];
    return color_map[2];
}


vec3f get_color_level0(int level){
    vec3f color_map[] = {
        get_color_by_hex(0x5e4fa2), get_color_by_hex(0x3288bd),
        get_color_by_hex(0x66c2a5), get_color_by_hex(0xabdda4),
        get_color_by_hex(0xe6f598), get_color_by_hex(0xffffbf),
        get_color_by_hex(0xfee08b), get_color_by_hex(0xfdae61),
        get_color_by_hex(0xf46d43), get_color_by_hex(0xd53e4f),
        get_color_by_hex(0x9e0142)
    };
    
    level -= 1;
    int count = sizeof(color_map) / sizeof(vec3f);
    if(level >= count) level = level % count;
    return level < count ? color_map[level] : vec3f(1,0,0);
}

void set_colors_temperature(float *col, SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    int count = pSet->GetParticleCount();
    Float Tmin = data->Tmin;
    Float Tmax = data->Tmax;
    int maxLevel = data->domain->GetLNMMaxLevel();
    for(int i = 0; i < count; i++){
        Float Ti = pSet->GetParticleTemperature(i);
        int level = (int)Floor((Tmax - Tmin) * (Ti - Tmin) / (Float)maxLevel);
        vec3f color = get_color_level(level);
        col[3 * i + 0] = color[0];
        col[3 * i + 1] = color[1];
        col[3 * i + 2] = color[2];
    }
}

void update_colors_cnm(float *col, SphSolverData2 *data){
    Grid2 *grid = data->domain;
    int count = grid->GetCellCount();
    for(int i = 0; i < count; i++){
        Cell2 *cell = grid->GetCell(i);
        int size = cell->GetChainLength();
        ParticleChain *pChain = cell->GetChain();
        for(int j = 0; j < size; j++){
            vec3f color = get_color_level(cell->GetLevel());
            col[3 * pChain->pId + 0] = color[0];
            col[3 * pChain->pId + 1] = color[1];
            col[3 * pChain->pId + 2] = color[2];
            pChain = pChain->next;
        }
    }
}

Float mapTo(Float A, Float B, Float a, Float b, Float x){
    Float BA = B - A;
    if(IsZero(BA)) return a;
    return (x - A) * (b - a) / BA + a;
}

void set_colors_pressure(float *col, SphSolverData2 *data){
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    Float pMin = FLT_MAX;
    Float pMax = -FLT_MAX;
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        Float pi = pSet->GetParticlePressure(i);
        pMin = Min(pi, pMin);
        pMax = Max(pi, pMax);
    }
    
    Float a = 0.3;
    
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        Float pi = pSet->GetParticlePressure(i);
        Float mpi = mapTo(pMin, pMax, a, 1, pi);
        col[3 * i + 0] = mpi;
        col[3 * i + 1] = mpi;
        col[3 * i + 2] = mpi;
    }
}

void set_colors_cnm(float *col, SphSolverData2 *data, int is_first, int classify){
    if(classify){
        if(is_first){
            UpdateGridDistributionGPU(data);
        }
        
        int level = LNMClassifyLazyGPU(data->domain);
        printf("Domain #levels: %d\n", level);
    }
    
    ParticleSet2 *pSet = data->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        int level = 0;
        if(classify){
            vec2f pi = pSet->GetParticlePosition(i);
            unsigned int id = data->domain->GetLinearHashedPosition(pi);
            Cell2 *cell = data->domain->GetCell(id);
            level = cell->GetLevel();
        }else{
            level = pSet->GetParticleV0(i);
        }
        
        vec3f color = get_color_level(level);
        col[3 * i + 0] = color[0];
        col[3 * i + 1] = color[1];
        col[3 * i + 2] = color[2];
    }
}

int set_poscol_cnm(float *col, float *pos, SphSolverData3 *data, 
                   int is_first, int classify)
{
    int level = 0;
    int it = 0;
    if(classify){
        if(is_first){
            UpdateGridDistributionGPU(data);
            (void)LNMClassifyLazyGPU(data->domain);
        }
        
        level = data->domain->GetLNMMaxLevel();
    }
    
    ParticleSet3 *pSet = data->sphpSet->GetParticleSet();
    for(int i = 0; i < pSet->GetParticleCount(); i++){
        vec3f pi = pSet->GetParticlePosition(i);
        level = 0;
        if(classify){
            unsigned int id = data->domain->GetLinearHashedPosition(pi);
            Cell3 *cell = data->domain->GetCell(id);
            level = cell->GetLevel();
        }else{
            level = pSet->GetParticleV0(i);
        }
        
        
        vec3f color = get_color_level(level);
        if(!(level == 1 || level == 2)){
            color = vec3f(0.7);
        }
        
        //if(pi.z < -0.65) continue;
        
        pos[3 * it + 0] = pi.x;
        pos[3 * it + 1] = pi.y;
        pos[3 * it + 2] = pi.z;
        col[3 * it + 0] = color[0];
        col[3 * it + 1] = color[1];
        col[3 * it + 2] = color[2];
        it += 1;
        
    }
    
    return it;
}

void test_sph2_double_dam_break(){
    printf("===== SPH Solver 2D -- Double Dam Break\n");
    Float spacing = 0.02;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    Grid2 *grid = cudaAllocateVx(Grid2, 1);
    SphSolver2 *solver = cudaAllocateVx(SphSolver2, 1);
    
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
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    memset(col, 0x00, sizeof(float) * 3 * count);
    SphSolverData2 *data = solver->GetSphSolverData();
    set_colors_cnm(col, data);
    
    for(int i = 0; i < 20 * 26; i++){
        solver->Advance(targetInterval);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_sph2_water_drop(){
    printf("===== SPH Solver 2D -- Water Drop\n");
    Float spacing = 0.02;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    Float r = 0.2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    Grid2 *grid = cudaAllocateVx(Grid2, 1);
    SphSolver2 *solver = cudaAllocateVx(SphSolver2, 1);
    
    int reso = (int)std::floor(lenc / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    Float boxLenx = lenc - 2 * spacing;
    Float boxLeny = 0.3;
    vec2f boxDim(boxLenx, boxLeny);
    
    solver->Initialize(DefaultSphSolverData2());
    Shape2 *rect = MakeRectangle2(Translate2(0, -lenc/2.f + boxLeny/2.f + spacing), boxDim);
    Shape2 *sphere = MakeSphere2(Translate2(0, lenc/2.f - r - spacing), r);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    grid->Build(res, pMin, pMax);
    
    VolumeParticleEmitterSet2 emitterSet;
    
    emitterSet.AddEmitter(sphere, sphere->GetBounds(), spacing);
    emitterSet.AddEmitter(rect, rect->GetBounds(), spacing);
    
    emitterSet.Emit(&builder);
    
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver->Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver->SetColliders(collider);
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver->GetSphSolverData();
    set_colors_cnm(col, data);
    
    for(int i = 0; i < 20 * 26; i++){
        solver->Advance(targetInterval);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_sph2_water_block(){
    printf("===== SPH Solver 2D -- Water Block\n");
    Float spacing = 0.03;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float lenc = 2;
    
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    
    Grid2 *grid = cudaAllocateVx(Grid2, 1);
    SphSolver2 *solver = cudaAllocateVx(SphSolver2, 1);
    
    int reso = (int)std::floor(lenc / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    solver->Initialize(DefaultSphSolverData2());
    Shape2 *rect = MakeRectangle2(Translate2(center.x, center.y+0.45), vec2f(1));
    Shape2 *block = MakeSphere2(Translate2(center.x, center.y-0.3), 0.2);
    Shape2 *container = MakeRectangle2(Translate2(center.x, center.y), vec2f(lenc), true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    grid->Build(res, pMin, pMax);
    
    VolumeParticleEmitter2 emitter(rect, rect->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(block);
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver->Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver->SetColliders(collider);
    
    Float targetInterval = 1.0 / 240.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver->GetSphSolverData();
    set_colors_cnm(col, data);
    
    for(int i = 0; i < 20 * 26; i++){
        solver->Advance(targetInterval);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_sph2_water_sphere(){
    printf("===== SPH Solver 2D -- Water in Ball\n");
    Float spacing = 0.015;
    Float targetDensity = WaterDensity;
    vec2f center(0,0);
    Float r1 = 0.5;
    Float r2 = 1.0;
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    Grid2 *grid = cudaAllocateVx(Grid2, 1);
    SphSolver2 *solver = cudaAllocateVx(SphSolver2, 1);
    
    int reso = (int)std::floor(2 * r2 / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    solver->Initialize(DefaultSphSolverData2());
    Shape2 *sphere = MakeSphere2(Translate2(center.x-0.4, center.y+r1/2.f), r1);
    //Shape2 *sphere = MakeSphere2(Translate2(center.x, center.y), r1);
    Shape2 *container = MakeSphere2(Translate2(center.x, center.y), r2, true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    grid->Build(res, pMin, pMax);
    
    VolumeParticleEmitter2 emitter(sphere, sphere->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2FromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver->Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver->SetColliders(collider);
    
    Float targetInterval = 1.0 / 248.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver->GetSphSolverData();
    set_colors_cnm(col, data);
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver->Advance(targetInterval);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}

void test_sph2_gas_sphere(){
    printf("===== SPH GAS Solver 2D -- Gas in Ball\n");
    Float spacing = 0.02;
    Float targetDensity = 1;
    vec2f center(0,0);
    Float r1 = 0.6;
    Float r2 = 1.0;
    vec2f pMin, pMax;
    Bounds2f containerBounds;
    ParticleSetBuilder2 builder;
    ColliderSetBuilder2 colliderBuilder;
    Grid2 *grid = cudaAllocateVx(Grid2, 1);
    SphGasSolver2 *solver = cudaAllocateVx(SphGasSolver2, 1);
    solver->Initialize();
    
    // NOTE: The higher the resolution the better for us
    //       since we can better explore GPU and our partitioning
    //       does not rely on sorting.
    int reso = (int)std::floor(2 * r2 / (spacing * 2.0));
    printf("Using grid with resolution %d x %d\n", reso, reso);
    vec2ui res(reso, reso);
    
    
    Shape2 *sphere = MakeSphere2(Translate2(center.x, center.y-r1 * 0.6), r1);
    Shape2 *container = MakeSphere2(Translate2(center.x, center.y), r2, true);
    
    containerBounds = container->GetBounds();
    pMin = containerBounds.pMin - vec2f(spacing);
    pMax = containerBounds.pMax + vec2f(spacing);
    
    grid->Build(res, pMin, pMax);
    
    VolumeParticleEmitter2 emitter(sphere, sphere->GetBounds(), spacing);
    
    emitter.Emit(&builder);
    SphParticleSet2 *sphSet = SphParticleSet2ExFromBuilder(&builder);
    ParticleSet2 *set2 = sphSet->GetParticleSet();
    int count = set2->GetParticleCount();
    
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    solver->Setup(targetDensity, spacing, 2.0, grid, sphSet);
    solver->SetColliders(collider);
    
    Float targetInterval = 1.0 / 2048.0;
    float *pos = new float[count * 3];
    float *col = new float[count * 3];
    
    SphSolverData2 *data = solver->GetSphSolverData();
    //set_colors_cnm(col, data);
    set_colors_temperature(col, data);
    for(int i = 0; i < 20 * 26 * 20; i++){
        solver->Advance(targetInterval);
        Debug_GraphyDisplaySolverParticles(sphSet->GetParticleSet(), pos, col);
    }
    
    delete[] pos;
    delete[] col;
    printf("===== OK\n");
}