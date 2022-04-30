#pragma once
#include <cutil.h>
#include <geometry.h>
#include <point_generator.h>
#include <kernel.h>
#include <set>

#define TimeStepLimitSpeedFactor 0.40
#define TimeStepLimitForceFactor 0.25
#define MaximumParticlesPerBucket 100

struct ParticleChain{
    unsigned int pId; // particle Id
    unsigned int cId; // cell Id
    unsigned int sId; // specie Id
    struct ParticleChain *next;
};

/*
* Buckets are the second degree optimization for cells.
*/
struct Bucket{
    int *pids;
    int size;
    int count;

    __host__ void SetSize(int n){
        size = n;
        pids = cudaAllocateVx(int, size);
    }

    __host__ void SetPointer(int *ptr, int n){
        pids = ptr;
        size = n;
        count = 0;
    }

    __bidevice__ int Count(){ return count; }

    __bidevice__ void Reset(){
        count = 0;
    }

    __bidevice__ void Insert(int pid){
        if(count < size){
            pids[count++] = pid;
        }else{
            //printf("Tried to insert without space (%d >= %d)\n", count, size);
        }
    }

    __bidevice__ int Get(int where){
        int r = -1;
        if(where < count){
            r = pids[where];
        }

        return r;
    }
};

/*
* Particle structure for ES-PIC solver
*/
template<typename T>
class SpecieSet{
    public:
    DataBuffer<T> positions;
    DataBuffer<T> velocities;
    DataBuffer<Float> mpWeight;
    Float mass;
    Float charge;

    DataBuffer<ParticleChain> chainNodes;
    DataBuffer<ParticleChain> chainAuxNodes;
    int count;
    unsigned int familyId;

    __bidevice__ SpecieSet(){ count = 0; }

    __bidevice__ void SetMass(Float m){ mass = m; }
    __bidevice__ Float GetMass(){ return mass; }
    __bidevice__ void SetCharge(Float ch){ charge = ch; }
    __bidevice__ Float GetCharge(){ return charge; }
    __bidevice__ void SetFamilyId(unsigned int id){ familyId = id; }
    __bidevice__ unsigned int GetFamilyId(){ return familyId; }

    __host__ void SetSize(int n){
        chainNodes.SetSize(n);
        chainAuxNodes.SetSize(n);
        positions.SetSize(n);
        velocities.SetSize(n);
        mpWeight.SetSize(n);
        count = 0;
    }

    __host__ void SetData(T *pos, T *vel, Float *mass, int n){
        positions.SetData(pos, n);
        velocities.SetData(vel, n);
        mpWeight.SetData(mass, n);
        chainNodes.SetSize(n);
        chainAuxNodes.SetSize(n);
        count = n;
    }

    __bidevice__ int GetParticleCount(){ return count; }
    __bidevice__ ParticleChain *GetParticleAuxChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle aux chain node");
        return chainAuxNodes.Get(pId);
    }

    __bidevice__ ParticleChain *GetParticleChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle chain node");
        return chainNodes.Get(pId);
    }

    __bidevice__ Float GetParticleMPW(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle mpw");
        return mpWeight.At(pId);
    }

    __bidevice__ T GetParticlePosition(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle position");
        return positions.At(pId);
    }

    __bidevice__ T GetParticleVelocity(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle velocity");
        return velocities.At(pId);
    }

    __bidevice__ void SetParticlePosition(int pId, const T &pos){
        AssertA(pId < count && pId >= 0, "Invalid set for particle position");
        positions.Set(pos, pId);
    }

    __bidevice__ void SetParticleVelocity(int pId, const T &vel){
        AssertA(pId < count && pId >= 0, "Invalid set for particle velocity");
        velocities.Set(vel, pId);
    }

    __bidevice__ void SetParticleMPW(int pId, Float mpw){
        AssertA(pId < count && pId >= 0, "Invalid set for particle MPW");
        mpWeight.Set(mpw, pId);
    }
};

typedef SpecieSet<vec2f> SpecieSet2;
typedef SpecieSet<vec3f> SpecieSet3;

/*
* Particle structure for SPH simulations
*/
template<typename T>
class ParticleSet{
    public:
    // Regular simulation data
    DataBuffer<T> positions;
    DataBuffer<T> velocities;
    DataBuffer<T> forces;
    DataBuffer<Float> pressures;
    DataBuffer<Float> densities;
    DataBuffer<T> normals;
    DataBuffer<Bucket> buckets;

    // Extended data for gaseous stuff
    DataBuffer<Float> temperature;
    DataBuffer<Float> densitiesEx;
    DataBuffer<Float> v0s;

    DataBuffer<ParticleChain> chainNodes;
    DataBuffer<ParticleChain> chainAuxNodes;
    int count;
    int familyId;

    Float radius;
    Float mass;

    __bidevice__ ParticleSet(){
        count = 0;
        radius = 1e-3;
        mass = 1e-3;
    }

    __bidevice__ int GetReservedSize(){
        return positions.GetSize();
    }

    __host__ void SetSize(int n){
        chainNodes.SetSize(n);
        chainAuxNodes.SetSize(n);
        positions.SetSize(n);
        velocities.SetSize(n);
        densities.SetSize(n);
        pressures.SetSize(n);
        forces.SetSize(n);
        normals.SetSize(n);
        v0s.SetSize(n);
        buckets.SetSize(n);
        radius = 1e-3;
        mass = 1e-3;
        familyId = 0;
        count = 0;
    }

    template<typename S> __host__ S *GetRawData(DataBuffer<S> buffer, int where){
        AssertA(where >= 0 && where < buffer.GetSize(), "Invalid raw data index");
        return buffer.Get(where);
    }

    __host__ void AppendData(T *pos, T *vel, T *force, int n){
        int rv = 0;
        rv |= positions.SetDataAt(pos, n, count);
        rv |= velocities.SetDataAt(vel, n, count);
        rv |= forces.SetDataAt(force, n, count);
        if(rv == 0){
            count += n;
        }
    }

    __host__ void SetData(T *pos, T *vel, T *force, int n)
    {
        positions.SetData(pos, n); velocities.SetData(vel, n);
        forces.SetData(force, n); densities.SetSize(n);
        pressures.SetSize(n); chainNodes.SetSize(n);
        chainAuxNodes.SetSize(n); v0s.SetSize(n);
        normals.SetSize(n); buckets.SetSize(n);
        count = n;
        radius = 1e-3;
        mass = 1e-3;
        familyId = 0;
    }

    __host__ void SetExtendedData(){
        densitiesEx.SetSize(count);
        temperature.SetSize(count);
    }

    template<typename F> __host__ void ClearDataBuffer(DataBuffer<F> *buffer){
        buffer->Clear();
    }

    __bidevice__ void SetRadius(Float rad){ radius = Max(0, rad); }
    __bidevice__ void SetMass(Float ms){ mass = Max(0, ms); }
    __bidevice__ Float GetRadius(){ return radius; }
    __bidevice__ Float GetMass(){ return mass; }
    __bidevice__ int GetParticleCount(){ return count; }
    __bidevice__ bool HasNormal(){ return normals.size > 0; }
    __bidevice__ void SetFamilyId(unsigned int id){ familyId = id; }
    __bidevice__ unsigned int GetFamilyId(){ return familyId; }

    __bidevice__ Bucket *GetParticleBucket(int pId){
        AssertA(pId < count && pId >= 0, "Invalid set for particle bucket");
        return buckets.Get(pId);
    }

    __bidevice__ ParticleChain *GetParticleAuxChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle aux chain node");
        return chainAuxNodes.Get(pId);
    }

    __bidevice__ ParticleChain *GetParticleChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle chain node");
        return chainNodes.Get(pId);
    }

    __bidevice__ Float GetParticleV0(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle v0");
        return v0s.At(pId);
    }

    __bidevice__ Float GetParticleTemperature(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle temperature");
        return temperature.At(pId);
    }

    __bidevice__ void SetParticleTemperature(int pId, Float temp){
        AssertA(pId < count && pId >= 0, "Invalid set for particle temperature");
        temperature.Set(temp, pId);
    }

    __bidevice__ T GetParticleNormal(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle normal");
        return normals.At(pId);
    }

    __bidevice__ Float GetParticleDensityEx(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle densityEx");
        return densitiesEx.At(pId);
    }

    __bidevice__ T GetParticlePosition(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle position");
        return positions.At(pId);
    }

    __bidevice__ T GetParticleVelocity(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle velocity");
        return velocities.At(pId);
    }

    __bidevice__ T GetParticleForce(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle forces");
        return forces.At(pId);
    }

    __bidevice__ Float GetParticleDensity(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle density");
        return densities.At(pId);
    }

    __bidevice__ Float GetParticlePressure(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle pressure");
        return pressures.At(pId);
    }

    __bidevice__ void SetParticleNormal(int pId, const T &nor){
        AssertA(pId < count && pId >= 0, "Invalid set for particle normal");
        normals.Set(nor, pId);
    }

    __bidevice__ void SetParticlePosition(int pId, const T &pos){
        AssertA(pId < count && pId >= 0, "Invalid set for particle position");
        positions.Set(pos, pId);
    }
    __bidevice__ void SetParticleVelocity(int pId, const T &vel){
        AssertA(pId < count && pId >= 0, "Invalid set for particle velocity");
        velocities.Set(vel, pId);
    }

    __bidevice__ void SetParticleForce(int pId, const T &force){
        AssertA(pId < count && pId >= 0, "Invalid set for particle forces");
        forces.Set(force, pId);
    }

    __bidevice__ void SetParticleDensity(int pId, Float density){
        AssertA(pId < count && pId >= 0, "Invalid set for particle density");
        densities.Set(density, pId);
    }

    __bidevice__ void SetParticlePressure(int pId, Float pressure){
        AssertA(pId < count && pId >= 0, "Invalid set for particle pressure");
        pressures.Set(pressure, pId);
    }

    __bidevice__ void SetParticleDensityEx(int pId, Float density){
        AssertA(pId < count && pId >= 0, "Invalid set for particle densityEx");
        densitiesEx.Set(density, pId);
    }

    __bidevice__ void SetParticleV0(int pId, Float v0){
        AssertA(pId < count && pId >= 0, "Invalid set for particle v0s");
        v0s.Set(v0, pId);
    }
};

typedef ParticleSet<vec2f> ParticleSet2;
typedef ParticleSet<vec3f> ParticleSet3;

class SphParticleSet2{
    public:
    ParticleSet2 *particleSet;
    Float targetDensity;
    Float targetSpacing;
    Float kernelRadiusOverTargetSpacing;
    Float kernelRadius;
    int requiresHigherLevelUpdate;

    __bidevice__ SphParticleSet2() : targetDensity(WaterDensity), targetSpacing(0.1),
    kernelRadiusOverTargetSpacing(2.0)
    {
        particleSet = nullptr;
    }

    __bidevice__ ParticleSet2 *GetParticleSet(){ return particleSet; }
    __bidevice__ Float GetKernelRadius(){ return kernelRadius; }
    __bidevice__ Float GetTargetDensity(){ return targetDensity; }
    __bidevice__ Float GetTargetSpacing(){ return targetSpacing; }
    __bidevice__ void SetHigherLevel(){ requiresHigherLevelUpdate = 1; }
    __bidevice__ void ResetHigherLevel(){ requiresHigherLevelUpdate = 0; }

    __bidevice__ void SetParticleData(ParticleSet2 *set){
        targetDensity = WaterDensity;
        targetSpacing = 0.1;
        kernelRadiusOverTargetSpacing = 2.0;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        particleSet = set;
    }

    __bidevice__ void SetTargetSpacing(Float spacing){
        AssertA(particleSet, "Invalid call to SphParticleSet::SetTargetSpacing");
        particleSet->SetRadius(spacing);
        targetSpacing = spacing;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    __bidevice__ void SetTargetDensity(Float density){
        targetDensity = density;
        ComputeMass();
    }

    __bidevice__ void SetRelativeKernelRadius(Float relativeRadius){
        kernelRadiusOverTargetSpacing = relativeRadius;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    __bidevice__ void ComputeMass(){
        int max_points = 128;
        vec2f points[128];
        TrianglePointGeneratorDevice pGenerastor;
        AssertA(!IsZero(kernelRadius), "Zero radius for mass computation");
        SphStdKernel2 kernel(kernelRadius);
        Bounds2f bounds(vec2f(-1.5 * kernelRadius), vec2f(1.5 * kernelRadius));
        Float nDen = 0;

        int c = pGenerastor.Generate(bounds, targetSpacing, &points[0], max_points);
        AssertA(c > 0, "Generated zero points");
        for(int i = 0; i < c; i++){
            vec2f pi = points[i];
            Float sum = 0;
            for(int j = 0; j < c; j++){
                vec2f pj = points[j];
                sum += kernel.W((pi - pj).Length());
            }

            nDen = Max(nDen, sum);
        }

        AssertA(!IsZero(nDen), "Zero number density");
        Float mass = targetDensity / nDen;
        particleSet->SetMass(mass);
    }

    __bidevice__ unsigned int ComputeNumberOfTimeSteps(Float timeStep, Float speedOfSound,
                                                       Float timeStepScale = 1.0)
    {
        AssertA(particleSet, "Invalid ParticleSet pointer for ComputeNumberOfTimeSteps");
        int count = particleSet->GetParticleCount();
        Float mass = particleSet->GetMass();
        Float maxForce = 0.0;

        for(int i = 0; i < count; i++){
            vec2f fi = particleSet->GetParticleForce(i);
            maxForce = Max(maxForce, fi.Length());
        }

        Float timeStepLimit = TimeStepLimitSpeedFactor * kernelRadius / speedOfSound;

        if(!IsZero(maxForce)){
            Float timeStepLimitbyForce =
                TimeStepLimitForceFactor * sqrt(kernelRadius * mass / maxForce);
            timeStepLimit = Min(timeStepLimitbyForce, timeStepLimit);
        }

        Float targetTimeStep = timeStepScale * timeStepLimit;
        return (unsigned int)std::ceil(timeStep / targetTimeStep);
    }
};

class SphParticleSet3{
    public:
    ParticleSet3 *particleSet;
    Float targetDensity;
    Float targetSpacing;
    Float kernelRadiusOverTargetSpacing;
    Float kernelRadius;
    int requiresHigherLevelUpdate;

    __bidevice__ SphParticleSet3() : targetDensity(WaterDensity), targetSpacing(0.1),
    kernelRadiusOverTargetSpacing(2.0)
    {
        particleSet = nullptr;
    }

    __bidevice__ ParticleSet3 *GetParticleSet(){ return particleSet; }
    __bidevice__ Float GetKernelRadius(){ return kernelRadius; }
    __bidevice__ Float GetTargetDensity(){ return targetDensity; }
    __bidevice__ Float GetTargetSpacing(){ return targetSpacing; }
    __bidevice__ void SetHigherLevel(){ requiresHigherLevelUpdate = 1; }
    __bidevice__ void ResetHigherLevel(){ requiresHigherLevelUpdate = 0; }

    __bidevice__ void SetParticleData(ParticleSet3 *set){
        targetDensity = WaterDensity;
        targetSpacing = 0.1;
        kernelRadiusOverTargetSpacing = 2.0;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        particleSet = set;
    }

    __bidevice__ void SetTargetSpacing(Float spacing){
        AssertA(particleSet, "Invalid call to SphParticleSet::SetTargetSpacing");
        particleSet->SetRadius(spacing);
        targetSpacing = spacing;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    __bidevice__ void SetTargetDensity(Float density){
        targetDensity = density;
        ComputeMass();
    }

    __bidevice__ void SetRelativeKernelRadius(Float relativeRadius){
        kernelRadiusOverTargetSpacing = relativeRadius;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    __bidevice__ void ComputeMass(){
        int max_points = 1024;
        vec3f points[1024];
        BccLatticePointGeneratorDevice pGenerator;
        AssertA(!IsZero(kernelRadius), "Zero radius for mass computation");
        SphStdKernel3 kernel(kernelRadius);
        Bounds3f bounds(vec3f(-1.5 * kernelRadius), vec3f(1.5 * kernelRadius));

        Float nDen = 0;
        int c = pGenerator.Generate(bounds, targetSpacing, &points[0], max_points);
        AssertA(c > 0, "Generated zero points");
        for(int i = 0; i < c; i++){
            vec3f pi = points[i];
            Float sum = 0;
            for(int j = 0; j < c; j++){
                vec3f pj = points[j];
                sum += kernel.W((pi - pj).Length());
            }

            nDen = Max(nDen, sum);
        }

        AssertA(!IsZero(nDen), "Zero number density");
        Float mass = targetDensity / nDen;
        particleSet->SetMass(mass);
    }

    __bidevice__ unsigned int ComputeNumberOfTimeSteps(Float timeStep, Float speedOfSound,
                                                       Float timeStepScale = 1.0)
    {
        AssertA(particleSet, "Invalid ParticleSet pointer for ComputeNumberOfTimeSteps");
        int count = particleSet->GetParticleCount();
        Float mass = particleSet->GetMass();
        Float maxForce = 0.0;
        Float refDist = kernelRadius;

        for(int i = 0; i < count; i++){
            vec3f fi = particleSet->GetParticleForce(i);
            maxForce = Max(maxForce, fi.Length());
        }

        Float timeStepLimit = TimeStepLimitSpeedFactor * refDist / speedOfSound;

        if(!IsZero(maxForce)){
            Float timeStepLimitbyForce =
                TimeStepLimitForceFactor * sqrt(refDist * mass / maxForce);
            timeStepLimit = Min(timeStepLimitbyForce, timeStepLimit);
        }

        Float targetTimeStep = timeStepScale * timeStepLimit;
        return (unsigned int)std::ceil(timeStep / targetTimeStep);
    }
};

template<typename T>
class ParticleSetBuilder{
    public:
    std::vector<T> positions;
    std::vector<T> velocities;
    std::vector<T> forces;
    std::vector<Float> mpw;

    std::vector<ParticleSet<T> *> sets;
    std::vector<SpecieSet<T> *> ssets;
    __host__ ParticleSetBuilder(){}

    __host__ void SetVelocityForAll(const T &vel){
        for(int i = 0; i < positions.size(); i++){
            velocities[i] = vel;
        }
    }

    __host__ int AddParticle(const T &pos, const T &vel = T(0),
                             const T &force = T(0))
    {
        positions.push_back(pos);
        velocities.push_back(vel);
        forces.push_back(force);
        return 1;
    }

    __host__ int AddParticle(const T &pos, Float mpW, const T &vel = T(0)){
        positions.push_back(pos);
        velocities.push_back(vel);
        mpw.push_back(mpW);
        return 1;
    }

    __host__ void Commit(){}

    __host__ int GetParticleCount(){ return positions.size(); }

    __host__ SpecieSet<T> * MakeSpecieSet(Float mass, Float charge){
        AssertA(positions.size() > 0, "No particles in builder");
        int cp = positions.size();
        int cv = velocities.size();
        int cw = mpw.size();
        AssertA(cp == cv && cp == cw, "Invalid particle configuration");

        SpecieSet<T> *pSet = cudaAllocateVx(SpecieSet<T>, 1);

        pSet->SetData(positions.data(), velocities.data(), mpw.data(), cp);
        pSet->SetMass(mass);
        pSet->SetCharge(charge);

        positions.clear();
        velocities.clear();
        mpw.clear();
        ssets.push_back(pSet);
        return pSet;
    }

    __host__ ParticleSet<T> * MakeExtendedParticleSet(){
        AssertA(positions.size() > 0, "No particles in builder");
        int cp = positions.size();
        int cv = velocities.size();
        int cf = forces.size();
        AssertA(cp == cv && cv == cf, "Invalid particle configuration");

        ParticleSet<T> *pSet = cudaAllocateVx(ParticleSet<T>, 1);
        pSet->SetData(positions.data(), velocities.data(), forces.data(), cp);
        pSet->SetExtendedData();

        positions.clear();
        velocities.clear();
        forces.clear();
        sets.push_back(pSet);
        return pSet;
    }

    __host__ ParticleSet<T> * MakeParticleSet(){
        AssertA(positions.size() > 0, "No particles in builder");
        int cp = positions.size();
        int cv = velocities.size();
        int cf = forces.size();
        AssertA(cp == cv && cv == cf, "Invalid particle configuration");

        ParticleSet<T> *pSet = cudaAllocateVx(ParticleSet<T>, 1);
        pSet->SetData(positions.data(), velocities.data(), forces.data(), cp);

        int pSize = MaximumParticlesPerBucket;
        int *ids = cudaAllocateVx(int, pSize * cp);

        int *ref = ids;
        for(int i = 0; i < cp; i++){
            Bucket *bucket = pSet->GetParticleBucket(i);
            bucket->SetPointer(&ref[i * pSize], pSize);
        }

        positions.clear();
        velocities.clear();
        forces.clear();
        sets.push_back(pSet);
        return pSet;
    }
};

typedef ParticleSetBuilder<vec2f> ParticleSetBuilder2;
typedef ParticleSetBuilder<vec3f> ParticleSetBuilder3;

__host__ SphParticleSet2 *SphParticleSet2FromBuilder(ParticleSetBuilder2 *builder);
__host__ SphParticleSet2 *SphParticleSet2ExFromBuilder(ParticleSetBuilder2 *builder);
__host__ SpecieSet2 *SpecieSet2FromBuilder(ParticleSetBuilder2 *builder,
                                           Float mass, Float charge, int familyId = 0);

__host__ SphParticleSet3 *SphParticleSet3FromBuilder(ParticleSetBuilder3 *builder);
