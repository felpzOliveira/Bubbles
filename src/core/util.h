#pragma once
#include <vector>
#include <geometry.h>
#include <transform.h>
#include <grid.h>
#include <emitter.h>
#include <collider.h>
#include <graphy.h>
#include <functional>
#include <statics.h>
#include <serializer.h>
#include <sph_solver.h>
#include <sstream>
#include <gDel3D/GpuDelaunay.h>
#include <gDel3D/CPU/PredWrapper.h>
#include <gDel3D/CommonTypes.h>

#if defined(DEBUG)
    #define BB_MSG(name) printf("* %s - Built %s at %s [DEBUG] *\n", name, __DATE__, __TIME__)
#else
    #define BB_MSG(name) printf("* %s - Built %s at %s *\n", name, __DATE__, __TIME__)
#endif

// Structures for hashing triangles for delaunay
template<int N>
struct iN{
    public:
    int t[N];

    iN(){
        for(int i = 0; i < N; i++) t[i] = 0;
    }

    iN(std::vector<int> f){
        bool sorting = false;
        if(f.size() != N){
            printf("INVALID INITIALIZATION\n");
        }else{
            for(int i = 0; i < N; i++){
                t[i] = f[i];
            }
        }

        for(int i = 0; i < N-1; i++){
            for(int j = 0; j < N - i - 1; j++){
                if(t[j] > t[j+1]){
                    Swap(t[j], t[j+1]);
                }
            }
        }
    }
};


template<int N> struct iNHasher{
    public:
    size_t operator()(const iN<N> &a) const{
        int f = 0;
        for(int i = 0; i < N; i++){
            f += a.t[i];
        }
        return std::hash<int>()(f);
    }
};

template<int N> struct iNIsSame{
    public:
    bool operator()(const iN<N> &a, const iN<N> &b) const{
        for(int i = 0; i < N; i++){
            if(b.t[i] != a.t[i]) return false;
        }
        return true;
    }
};


struct i3{
    public:
    int t[3];

    i3(){ t[0] = 0; t[1] = 0; t[2] = 0; }

    i3(int a, int b, int c){
        if(a > c) Swap(a, c);
        if(a > b) Swap(a, b);
        if(b > c) Swap(b, c);
        t[0] = a;
        t[1] = b;
        t[2] = c;
    }
};


struct i3Hasher{
    public:
    size_t operator()(const i3 &a) const{
        int f = a.t[0] + a.t[1] + a.t[2];
        return std::hash<int>()(f);
    }
};

struct i3IsSame{
    public:
    bool operator()(const i3 &a, const i3 &b) const{
        return b.t[0] == a.t[0] && b.t[1] == a.t[1] && b.t[2] == a.t[2];
    }
};

/*
* So... our simulator is looking great! However is very hard to perform
* full setup of a new scene, let's add some funcionality to help with that
* and check for common error conditions that will trigger several issues
* and make us waste time debugging.
*/

/*
* Get a set of scattered particles for displaying the FieldGrid SDF.
* Returns the amount of particles added to the particles vector.
*/
__host__ int UtilGetSDFParticles(FieldGrid3f *field, std::vector<vec3f> *particles,
                                 Float sdfThreshold, Float spacing, int absolute=1);

/*
* Computes the bounds of a mesh by inspecting all of its vertex.
*/
__host__ Bounds3f UtilComputeMeshBounds(ParsedMesh *mesh);

/*
* Computes a scale transform that makes sure the given mesh fits in a maximum length.
*/
__host__ Transform UtilComputeFitTransform(ParsedMesh *mesh, Float maximumAxisLength,
                                           Float *scaleValue=nullptr);

/*
* Computes the bounds of a mesh after a transformation.
*/
__host__ Bounds3f UtilComputeBoundsAfter(ParsedMesh *mesh, Transform transform);


/*
* Generates a acceleration Grid for a domain given its bounds, the target spacing
* of the simulation and the spacing scale to be used. This grid is uniform.
*/
__host__ Grid3 *UtilBuildGridForDomain(Bounds3f domain, Float spacing,
                                       Float spacingScale = 2.0);

__host__ Grid2 *UtilBuildGridForDomain(Bounds2f domain, Float spacing,
                                       Float spacingScale = 2.0);

__host__ Grid3 *UtilBuildGridForBuilder(ParticleSetBuilder3 *builder,
                                        Float spacing, Float spacingScale);

/*
* Checks if emitting from any of the emitters in VolumeParticleEmitterSet3 
* will overlaps any of the colliders given in the ColliderSet3. This is important
* as distributing inside a collider that doesn't have reverseOrientation=true will
* cause a full redistribution by our grid hashing scheme and may generate out of bounds
* particles.
*/
__host__ int UtilIsEmitterOverlapping(VolumeParticleEmitter3 *emitterSet,
                                      ColliderSet3 *colliderSet);

__host__ int UtilIsEmitterOverlapping(VolumeParticleEmitterSet3 *emitterSet,
                                      ColliderSet3 *colliderSet);

/*
* Parses a BB file and add its particles to a ParticleSetBuilder3 builder.
* You can transform (rotate, scale) the input data with a transform and can
* translate the dataset to be around 'centerAt'. This routine applies a translation
* to centerAt after the given transform. You can also set the initial velocity vector
* by using the 'initialVelocity' parameter.
* Returns the bounds taken by transformed particles.
*/
__host__ Bounds3f UtilParticleSetBuilder3FromBB(const char *path, ParticleSetBuilder3 *builder,
                                                int legacy=0, Transform transform=Transform(),
                                                vec3f centerAt=vec3f(0),
                                                vec3f initialVelocity=vec3f(0));

/*
* Creates a SDF on field for the given particle distribution in the Sph Solver using
* density cutOffDensity - density as node value.
*/
__host__ void UtilSphDataToFieldGrid2f(SphSolverData2 *solverData, FieldGrid2f *field);

/*
* Generates points around a circle of radius 'rad' uniformly settings color to 'col'.
*/
__host__ int UtilGenerateCirclePoints(float *posBuffer, float *colBuffer, vec3f col,
                                      vec2f center, Float rad, int nPoints);

/*
* Generates points around a square of size 'len'. The given transform is used to move
* points to a specific location as the square is generated around the origin.
*/
__host__ int UtilGenerateSquarePoints(float *posBuffer, float *colBuffer, vec3f col,
                                      Transform2 transform, vec2f len, int nPoints);

/*
* Generates points around a sphere of radius 'rad' uniformly settings color to 'col'.
*/
__host__ int UtilGenerateSpherePoints(float *posBuffer, float *colBuffer, vec3f col,
                                      Float rad, int nPoints, Transform transform);

/*
* Generates points around a box of size 'length' settings colors to 'col'.
*/
__host__ int UtilGenerateBoxPoints(float *posBuffer, float *colBuffer, vec3f col,
                                   vec3f length, int nPoints, Transform transform);

/*
* Writes the output of GDel3D to a ply file.
*/
__host__ void UtilGDel3DWritePly(Point3HVec *pointVec, GDelOutput *output, int pLen,
                                 const char *path, bool tetras=true);

/*
* Writes a ply file from a specific set of triangles from a given GDel3D output.
*/
__host__ void UtilGDel3DWritePly(std::vector<i3> *tris, Point3HVec *pointVec,
                                 GDelOutput *output, const char *path);

template<typename Fn>
__host__ void GDel3D_ForEachRealTetra(GDelOutput *output, uint32_t pLen, Fn fn){
    const TetHVec tetVec      = output->tetVec;
    const TetOppHVec oppVec   = output->tetOppVec;
    const CharHVec tetInfoVec = output->tetInfoVec;

    for(int i = 0; i < tetVec.size(); i++){
        bool valid = true;
        Tet tet = tetVec[i];
        const TetOpp botOpp = oppVec[i];
        if(!isTetAlive(tetInfoVec[i])) valid = false;

        for(int s = 0; s < 4; s++){
            if(botOpp._t[s] == -1) valid = false;
            if(tet._v[s] == pLen) valid = false; // inf point
        }

        if(valid)
            fn(tet, botOpp, i);
    }
}


/*
* Find all triangles that are unique.
*/
__host__ void UtilGDel3DUniqueTris(std::vector<i3> &tris, Point3HVec *pointVec,
                                   GDelOutput *output, int pLen);

template<typename T, typename U, typename Q> inline __host__
int UtilIsDistributionConsistent(ParticleSet<T> *pSet, Grid<T, U, Q> *grid){
    // for now just check all hashes match
    Q gridBounds = grid->GetBounds();

    for(int i = 0; i < pSet->GetParticleCount(); i++){
        T p = pSet->GetParticlePosition(i);
        if(!Inside(p, gridBounds)){
            T p0 = gridBounds.pMin;
            T p1 = gridBounds.pMax;
            if(grid->GetDimensions() == 3){
                printf("**************************************\n");
                printf("Warning: Domain {%g %g %g} x {%g %g %g} "
                       "cannot hash position {%g %g %g}\n", p0[0], p0[1], p0[2],
                       p1[0], p1[1], p1[2], p[0], p[1], p[2]);
                printf("**************************************\n");
            }else{
                printf("**************************************\n");
                printf("Warning: Domain {%g %g} x {%g %g} "
                       "cannot hash position {%g %g}\n", p0[0], p0[1],
                       p1[0], p1[1], p[0], p[1]);
                printf("**************************************\n");
            }
            return 0;
        }
    }

    return 1;
}

inline
__host__ int UtilIsDomainContaining(Bounds3f domainBounds, std::vector<Bounds3f> testBounds)
{
    vec3f center = domainBounds.Center();
    for(int i = 0; i < testBounds.size(); i++){
        Bounds3f bound = testBounds[i];
        vec3f otherCenter = bound.Center();
        int rv = -1;
        for(int i = 0; i < 3; i++){
            Float S = 0, s = 0;
            S = center[i] + domainBounds.ExtentOn(i) * 0.5;
            s = otherCenter[i] + bound.ExtentOn(i) * 0.5;
            if(S < s) { rv = i; break; }

            S = center[i] - domainBounds.ExtentOn(i) * 0.5;
            s = otherCenter[i] - bound.ExtentOn(i) * 0.5;
            if(S > s) { rv = i; break; }
        }

        if(rv != -1){
            vec3f p0 = domainBounds.pMin;
            vec3f p1 = domainBounds.pMax;
            vec3f v0 = bound.pMin;
            vec3f v1 = bound.pMax;
            printf("*********************************************************\n");
            printf("Warning: Domain {%g %g %g} x {%g %g %g} does not contain:\n"
                   "\t{%g %g %g} x {%g %g %g}\n", p0.x, p0.y, p0.z, p1.x, p1.y, p1.z,
                   v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
            printf("Offending axis: %d\n", rv);
            printf("*********************************************************\n");
            return 0;
        }
    }

    return 1;
}

/*
* Compute the bounds of a given particle set.
*/
inline
__host__ Bounds3f UtilComputeParticleSetBounds(ParticleSet3 *pSet){
    int count = pSet->GetParticleCount();
    vec3f pi = pSet->GetParticlePosition(0);
    Bounds3f bounds(pi, pi);
    for(int i = 1; i < count; i++){
        pi = pSet->GetParticlePosition(i);
        bounds = Union(bounds, pi);
    }

    return bounds;
}

inline
__host__ Bounds2f UtilComputeParticleSetBounds(ParticleSet2 *pSet){
    int count = pSet->GetParticleCount();
    vec2f pi = pSet->GetParticlePosition(0);
    Bounds2f bounds(pi, pi);
    for(int i = 1; i < count; i++){
        pi = pSet->GetParticlePosition(i);
        bounds = Union(bounds, pi);
    }

    return bounds;
}

template<typename DataAccessor>
inline __host__ Float UtilComputeMedian(DataAccessor *accessor, int size){
    double value = 0;
    for(int i = 0; i < size; i++){
        value += accessor[i];
    }

    value /= (double)size;
    return (Float)value;
}

template<typename ParticleAccessor>
inline __host__ int UtilFillBoundaryParticles(ParticleAccessor *pSet,
                                              std::vector<int> *boundaries)
{
    int bCount = 0;
    int pCount = pSet->GetParticleCount();
    boundaries->clear();

    for(int i = 0; i < pCount; i++){
        int L = pSet->GetParticleV0(i);
        if(L > 0){
            boundaries->push_back(L);
            bCount ++;
        }else{
            boundaries->push_back(0);
        }
    }

    return bCount;
}

inline __host__ void UtilEraseFile(const char *filename){
    remove(filename);
}

/*
* Saves a 3D simulation. This routine assumes that the last collider present
* in the solver is the domain and does not write it, only colliders that are
* obstacles are written. TODO: Split coliders/domain from the collider set builder?
*/
template<typename Solver, typename ParticleAccessor>
inline __host__ void UtilSaveSimulation3(Solver *solver, ParticleAccessor *pSet,
                                         const char *filename, int flags)
{
    std::stringstream ss;
    std::vector<int> boundaries;

    UtilEraseFile(filename);

    FILE *fp = fopen(filename, "a+");
    if(!fp){
        printf("Failed to open file %s\n", filename);
        return;
    }

    ColliderSet3 *colSet3 = solver->GetColliders();
    for(int i = 0; i < colSet3->nColiders-1; i++){
        if(colSet3->IsActive(i)){
            Collider3 *colider = colSet3->colliders[i];
            ss << colider->shape->Serialize() << std::endl;
        }
    }

    std::string str = ss.str();
    fprintf(fp, "%s", str.c_str());
    fclose(fp);

    UtilGetBoundaryState(pSet, &boundaries);
    SerializerSaveSphDataSet3(solver->GetSphSolverData(), filename, flags, &boundaries);
}

template<typename Solver, typename ParticleAccessor> inline __host__
void UtilSaveSimulation3(Solver *solver, std::vector<ParticleAccessor *> pSets,
                         const char *filename, int flags)
{
    std::stringstream ss;
    UtilEraseFile(filename);
    FILE *fp = fopen(filename, "a+");
    if(!fp){
        printf("Failed to open file %s\n", filename);
        return;
    }

    ColliderSet3 *colSet3 = solver->GetColliders();
    for(int i = 0; i < colSet3->nColiders-1; i++){
        if(colSet3->IsActive(i)){
            Collider3 *colider = colSet3->colliders[i];
            ss << colider->shape->Serialize() << std::endl;
        }
    }

    std::string str = ss.str();
    fprintf(fp, "%s", str.c_str());
    fclose(fp);

    flags &= ~SERIALIZER_BOUNDARY;
    SerializerSaveSphDataSet3Many(solver->GetSphSolverData(), pSets, filename, flags);
}

/*
* Run a 2D simulation. Perform several updates on the given solver and display results
* interactivily using graphy. View setup is made by the vectors 'lower' and 'upper'.
* You can save a frame or update emitors and calliders from the callback function which
* is called at the begining of every step.
* Callback should return 0 if simulation should stop or != 0 to continue.
*/
template<typename Solver, typename ParticleAccessor>
inline __host__ void UtilRunSimulation2(Solver *solver, ParticleAccessor *pSet,
                                        Float spacing, vec2f lower, vec2f upper,
                                        Float targetInterval,
                                        const std::function<int(int )> &callback)
{
    float *ptr = nullptr;
    float *pos = nullptr;
    float *col = nullptr;
    float pSize = 2.0f;
    int visible = 0;
    int total = pSet->GetReservedSize();
    ptr = new float[2 * 3 * total];
    pos = &ptr[0];
    col = &ptr[3 * total];

    memset(col, 0, sizeof(float) * 3 * total);
    visible = pSet->GetParticleCount();
    for(int j = 0; j < visible; j++){
        vec2f pi = pSet->GetParticlePosition(j);
        pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
        pos[3 * j + 2] = 0;    col[3 * j + 0] = 1;
    }


    graphy_render_points_size(pos, col, pSize, visible,
                              lower.x, upper.x, upper.y, lower.y);
    int frame = 0;
    while(callback(frame) != 0){
        solver->Advance(targetInterval);
        visible = pSet->GetParticleCount();
        for(int j = 0; j < visible; j++){
            vec2f pi = pSet->GetParticlePosition(j);
            pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
            pos[3 * j + 2] = 0;    col[3 * j + 0] = 1;
        }
        graphy_render_points_size(pos, col, pSize, visible,
                                  lower.x, upper.x, upper.y, lower.y);
        frame++;
    }

    graphy_close_display();
    delete[] ptr;
}

/*
* Run a 2D simulation. Perform several updates on the given solver and display results
* interactivily using graphy. View setup is made by the vectors 'lower' and 'upper'.
* You can save a frame or update emitors and calliders from the callback function which
* is called at the begining of every step. This call receives a color function to
* configure the display.
* Callback should return 0 if simulation should stop or != 0 to continue.
*/
template<typename Solver, typename ParticleAccessor>
inline __host__ void UtilRunSimulation2(Solver *solver, ParticleAccessor *pSet,
                                        Float spacing, vec2f lower, vec2f upper,
                                        Float targetInterval,
                                        const std::function<int(int )> &callback,
                                        const std::function<void(float *, int)> &setCol)
{
    float *ptr = nullptr;
    float *pos = nullptr;
    float *col = nullptr;
    float pSize = 2.0f;
    int visible = 0;
    int total = pSet->GetReservedSize();
    ptr = new float[2 * 3 * total];
    pos = &ptr[0];
    col = &ptr[3 * total];

    memset(col, 0, sizeof(float) * 3 * total);
    visible = pSet->GetParticleCount();
    for(int j = 0; j < visible; j++){
        vec2f pi = pSet->GetParticlePosition(j);
        pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
        pos[3 * j + 2] = 0;
    }

    setCol(col, visible);

    graphy_render_points_size(pos, col, pSize, visible,
                              lower.x, upper.x, upper.y, lower.y);
    int frame = 0;
    while(callback(frame) != 0){
        solver->Advance(targetInterval);
        visible = pSet->GetParticleCount();
        for(int j = 0; j < visible; j++){
            vec2f pi = pSet->GetParticlePosition(j);
            pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
            pos[3 * j + 2] = 0;
        }

        setCol(col, visible);
        graphy_render_points_size(pos, col, pSize, visible,
                                  lower.x, upper.x, upper.y, lower.y);
        frame++;
    }

    graphy_close_display();
    delete[] ptr;
}

/*
* Run a 2D simulation. Perform several updates on the given solver and display results
* interactivily using graphy. View setup is made by the vectors 'lower' and 'upper'.
* You can save a frame or update emitors and calliders from the callback function which
* is called at the begining of every step. This call receives a color function to
* configure the display. This function allows a callback to give extra inputs per frame,
* i.e.: you can use it to fill up to 'extraParts' of data to be displayed.
* Callback should return 0 if simulation should stop or != 0 to continue.
*/
template<typename Solver, typename ParticleAccessor>
inline __host__ void UtilRunDynamicSimulation2(Solver *solver, ParticleAccessor *pSet,
                                               Float spacing, vec2f lower, vec2f upper,
                                               Float targetInterval, int extraParts,
                                               const std::function<int(int )> &callback,
                                               const std::function<void(float *, int)> &setCol,
                                               const std::function<int(float*, float*)> &filler)
{
    float *ptr = nullptr;
    float *pos = nullptr;
    float *col = nullptr;
    float pSize = 2.0f;
    int visible = 0;
    int extra = 0;
    int total = pSet->GetReservedSize() + extraParts;
    ptr = new float[2 * 3 * total];
    pos = &ptr[0];
    col = &ptr[3 * total];

    memset(col, 0, sizeof(float) * 3 * total);
    visible = pSet->GetParticleCount();
    for(int j = 0; j < visible; j++){
        vec2f pi = pSet->GetParticlePosition(j);
        pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
        pos[3 * j + 2] = 0;
    }

    setCol(col, visible);
    extra = filler(&pos[visible * 3], &col[visible * 3]);

    graphy_render_points_size(pos, col, pSize, visible + extra,
                              lower.x, upper.x, upper.y, lower.y);
    int frame = 0;
    while(callback(frame) != 0){
        solver->Advance(targetInterval);
        visible = pSet->GetParticleCount();
        for(int j = 0; j < visible; j++){
            vec2f pi = pSet->GetParticlePosition(j);
            pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
            pos[3 * j + 2] = 0;
        }

        setCol(col, visible);
        extra = filler(&pos[visible * 3], &col[visible * 3]);
        graphy_render_points_size(pos, col, pSize, visible + extra,
                                  lower.x, upper.x, upper.y, lower.y);
        frame++;
    }

    graphy_close_display();
    delete[] ptr;
}

/*
* Run a 3D simulation. Perform several updates on the given solver and display results
* interactivily using graphy. Camera setup is made by the vectors 'origin' and 'target'.
* You can save a frame or update emitors and calliders from the callback function which
* is called at the begining of every step.
* Callback should return 0 if simulation should stop or != 0 to continue.
*/
template<typename Solver, typename ParticleAccessor>
inline __host__ void UtilRunSimulation3(Solver *solver, ParticleAccessor *pSet,
                                        Float spacing, vec3f origin, vec3f target,
                                        Float targetInterval, std::vector<Shape*> sdfs,
                                        const std::function<int(int )> &callback)
{
    std::vector<vec3f> particles;
    int total = pSet->GetReservedSize();
    float *ptr = nullptr;
    float *pos = nullptr;
    float *col = nullptr;
    for(Shape *shape : sdfs){
        UtilGetSDFParticles(shape->grid, &particles, 0, spacing, 0);
    }

    total += particles.size();
    ptr = new float[2 * 3 * total];
    pos = &ptr[0];
    col = &ptr[3 * total];

    memset(col, 0, sizeof(float) * 3 * total);

    int it = 0;
    for(int i = 0; i < particles.size(); i++){
        vec3f pi = particles[it++];
        pos[3 * i + 0] = pi.x; pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = pi.z; col[3 * i + 2] = 1;
    }

    int end = pSet->GetParticleCount();
    for(int i = 0; i < end; i++){
        vec3f pi = pSet->GetParticlePosition(i);
        int j = i + particles.size();
        pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
        pos[3 * j + 2] = pi.z; col[3 * j + 0] = 1;
    }


    graphy_set_3d(origin.x, origin.y, origin.z, target.x, target.y, target.z,
                  45.0, 0.1f, 100.0f);
    int visible = particles.size() + pSet->GetParticleCount();
    graphy_render_points3f(pos, col, visible, spacing/2.0);
    int frame = 0;

    while(callback(frame) != 0){
        solver->Advance(targetInterval);
        for(int j = 0; j < pSet->GetParticleCount(); j++){
            vec3f pi = pSet->GetParticlePosition(j);
            int k = j + particles.size();
            pos[3 * k + 0] = pi.x; pos[3 * k + 1] = pi.y;
            pos[3 * k + 2] = pi.z; col[3 * k + 0] = 1;
        }

        visible = particles.size() + pSet->GetParticleCount();
        graphy_render_points3f(pos, col, visible, spacing/2.0);
        frame++;
    }

    graphy_close_display();
    delete[] ptr;
}

/*
* Run a 3D simulation. Perform several updates on the given solver and display results
* interactivily using graphy. Camera setup is made by the vectors 'origin' and 'target'.
* You can save a frame or update emitors and calliders from the callback function which
* is called at the begining of every step.
* Callback should return 0 if simulation should stop or != 0 to continue.
*/
template<typename Solver, typename ParticleAccessor>
inline __host__ void UtilRunDynamicSimulation3(Solver *solver, ParticleAccessor *pSet,
                                               Float spacing, vec3f origin, vec3f target,
                                               Float targetInterval, int extraParts,
                                               std::vector<Shape*> sdfs,
                                               const std::function<int(int )> &callback,
                                               const std::function<void(float*,int)> &setCol,
                                               const std::function<int(float*,float*)> &filler)
{
    std::vector<vec3f> particles;
    int total = pSet->GetReservedSize();
    int sdfSize = 0;
    float *ptr = nullptr;
    float *pos = nullptr;
    float *col = nullptr;
    int extra = 0;
    for(Shape *shape : sdfs){
        UtilGetSDFParticles(shape->grid, &particles, 0, spacing);
    }

    sdfSize = particles.size();
    total += particles.size();
    total += extraParts;
    ptr = new float[2 * 3 * total];
    pos = &ptr[0];
    col = &ptr[3 * total];

    memset(col, 0, sizeof(float) * 3 * total);

    int it = 0;
    for(int i = 0; i < particles.size(); i++){
        vec3f pi = particles[it++];
        pos[3 * i + 0] = pi.x; pos[3 * i + 1] = pi.y;
        pos[3 * i + 2] = pi.z; col[3 * i + 2] = 1;
    }

    int end = pSet->GetParticleCount();
    for(int i = 0; i < end; i++){
        vec3f pi = pSet->GetParticlePosition(i);
        int j = i + sdfSize;
        pos[3 * j + 0] = pi.x; pos[3 * j + 1] = pi.y;
        pos[3 * j + 2] = pi.z;
    }

    setCol(&col[3 * sdfSize], end);
    extra = filler(&pos[3 * (sdfSize + end)], &col[3 * (sdfSize + end)]);

    graphy_set_3d(origin.x, origin.y, origin.z, target.x, target.y, target.z,
                  45.0, 0.1f, 100.0f);
    int visible = particles.size() + pSet->GetParticleCount() + extra;
    graphy_render_points3f(pos, col, visible, spacing/2.0);
    int frame = 0;

    while(callback(frame) != 0){
        solver->Advance(targetInterval);
        end = pSet->GetParticleCount();
        for(int j = 0; j < end; j++){
            vec3f pi = pSet->GetParticlePosition(j);
            int f = j + sdfSize;
            pos[3 * f + 0] = pi.x; pos[3 * f + 1] = pi.y;
            pos[3 * f + 2] = pi.z;
        }

        setCol(&col[3 * sdfSize], end);
        extra = filler(&pos[3 * (sdfSize + end)], &col[3 * (sdfSize + end)]);

        visible = particles.size() + end + extra;
        graphy_render_points3f(pos, col, visible, spacing/2.0);
        frame++;
    }

    graphy_close_display();
    delete[] ptr;
}

/*
* Print LNM stats after a step of simulation.
*/
template<typename Solver>
inline __host__ void UtilPrintStepStandard(Solver *solver, int step,
                                           std::vector<int> refFrames={})
{
    LNMData faster, slower, last, average;
    LNMStats stats = solver->GetLNMStats();
    Float advTime = solver->GetAdvanceTime();
    average = stats.Average(&faster, &slower);
    last = stats.Last();
    int pCount = solver->GetParticleCount();
    int takeFrame = 0;
    for(int &f : refFrames){
        if(step == f){
            takeFrame = 1;
            break;
        }
    }

    printf("\rStep (%d) : %d ms - Particles: %d - (LNM: %g ms %g%%) (Slower: %g ms %g%%) (Faster: %g ms %g%%)    ",
           step, (int)advTime, pCount, average.timeTaken, average.simPercentage,
           slower.timeTaken, slower.simPercentage, faster.timeTaken, faster.simPercentage);
    if(takeFrame) printf("\n");
}

/*
* Dumps the basic position of the current simulator step into disk.
*/
inline __host__ void UtilSaveSph3Frame(const char *basedir, int step, SphSolverData3 *data){
    std::string path(basedir);
    path += std::to_string(step);
    path += ".txt";
    SerializerSaveSphDataSet3(data, path.c_str(), SERIALIZER_POSITION);
}


/*
* Fetch the current boundary of the particle set.
*/
template<typename ParticleSetAccessor> inline __host__
int UtilGetBoundaryState(ParticleSetAccessor *pSet, std::vector<int> *boundaries){
    int count = pSet->GetParticleCount();
    int n = 0;
    boundaries->clear();
    for(int i = 0; i < count; i++){
        int b = 0;
        int v0 = pSet->GetParticleV0(i);
        if(v0 > 0){
            b = v0;
            n++;
        }

        boundaries->push_back(b);
    }

    return n;
}
