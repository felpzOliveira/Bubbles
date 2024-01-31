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

    void SetSize(int n){
        size = n;
        pids = cudaAllocateVx(int, size);
    }

    void SetPointer(int *ptr, int n){
        pids = ptr;
        size = n;
        count = 0;
    }

    bb_cpu_gpu int Count(){ return count; }

    bb_cpu_gpu void Reset(){
        count = 0;
    }

    bb_cpu_gpu void Insert(int pid){
        if(count < size){
            pids[count++] = pid;
        }else{
            //printf("Tried to insert without space (%d >= %d)\n", count, size);
        }
    }

    bb_cpu_gpu int Get(int where){
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

    bb_cpu_gpu SpecieSet(){ count = 0; }

    bb_cpu_gpu void SetMass(Float m){ mass = m; }
    bb_cpu_gpu Float GetMass(){ return mass; }
    bb_cpu_gpu void SetCharge(Float ch){ charge = ch; }
    bb_cpu_gpu Float GetCharge(){ return charge; }
    bb_cpu_gpu void SetFamilyId(unsigned int id){ familyId = id; }
    bb_cpu_gpu unsigned int GetFamilyId(){ return familyId; }

    void SetSize(int n){
        chainNodes.SetSize(n);
        chainAuxNodes.SetSize(n);
        positions.SetSize(n);
        velocities.SetSize(n);
        mpWeight.SetSize(n);
        count = 0;
    }

    void SetData(T *pos, T *vel, Float *mass, int n){
        positions.SetData(pos, n);
        velocities.SetData(vel, n);
        mpWeight.SetData(mass, n);
        chainNodes.SetSize(n);
        chainAuxNodes.SetSize(n);
        count = n;
    }

    bb_cpu_gpu int GetParticleCount(){ return count; }
    bb_cpu_gpu ParticleChain *GetParticleAuxChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle aux chain node");
        return chainAuxNodes.Get(pId);
    }

    bb_cpu_gpu ParticleChain *GetParticleChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle chain node");
        return chainNodes.Get(pId);
    }

    bb_cpu_gpu Float GetParticleMPW(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle mpw");
        return mpWeight.At(pId);
    }

    bb_cpu_gpu T GetParticlePosition(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle position");
        return positions.At(pId);
    }

    bb_cpu_gpu T GetParticleVelocity(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle velocity");
        return velocities.At(pId);
    }

    bb_cpu_gpu void SetParticlePosition(int pId, const T &pos){
        AssertA(pId < count && pId >= 0, "Invalid set for particle position");
        positions.Set(pos, pId);
    }

    bb_cpu_gpu void SetParticleVelocity(int pId, const T &vel){
        AssertA(pId < count && pId >= 0, "Invalid set for particle velocity");
        velocities.Set(vel, pId);
    }

    bb_cpu_gpu void SetParticleMPW(int pId, Float mpw){
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

    // TODO: other datatypes
    DataBuffer<vec3f> *userVec3Buffers;
    int userVec3Count;

    DataBuffer<ParticleChain> chainNodes;
    DataBuffer<ParticleChain> chainAuxNodes;
    int count;
    int familyId;

    Float radius;
    Float mass;

    bb_cpu_gpu ParticleSet(){
        count = 0;
        radius = 1e-3;
        mass = 1e-3;
        userVec3Buffers = nullptr;
    }

    bb_cpu_gpu int GetReservedSize(){
        return positions.GetSize();
    }

    void SetSize(int n){
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
        userVec3Buffers = nullptr;
        userVec3Count = 0;
    }

    void SetUserVec3Buffer(int n){
        n = Max(0, n);
        if(n == 0 || count == 0)
            return;

        userVec3Buffers = cudaAllocateVx(DataBuffer<vec3f>, n);
        for(int i = 0; i < n; i++)
            userVec3Buffers[i].SetSize(count);

        userVec3Count = n;
    }

    template<typename S> S *GetRawData(DataBuffer<S> buffer, int where){
        AssertA(where >= 0 && where < buffer.GetSize(), "Invalid raw data index");
        return buffer.Get(where);
    }

    void AppendData(T *pos, T *vel, T *force, int n){
        int rv = 0;
        rv |= positions.SetDataAt(pos, n, count);
        rv |= velocities.SetDataAt(vel, n, count);
        rv |= forces.SetDataAt(force, n, count);
        if(rv == 0){
            count += n;
        }
    }

    void SetData(T *pos, T *vel, T *force, int n)
    {
        positions.SetData(pos, n); velocities.SetData(vel, n);
        forces.SetData(force, n);  densities.SetSize(n);
        pressures.SetSize(n);      chainNodes.SetSize(n);
        chainAuxNodes.SetSize(n);  v0s.SetSize(n);
        normals.SetSize(n);        buckets.SetSize(n);
        count = n;
        radius = 1e-3;
        mass = 1e-3;
        familyId = 0;
        userVec3Buffers = nullptr;
        userVec3Count = 0;
    }

    void SetExtendedData(){
        densitiesEx.SetSize(count);
        temperature.SetSize(count);
    }

    template<typename F> void ClearDataBuffer(DataBuffer<F> *buffer){
        buffer->Clear();
    }

    bb_cpu_gpu void SetRadius(Float rad){ radius = Max(0, rad); }
    bb_cpu_gpu void SetMass(Float ms){ mass = Max(0, ms); }
    bb_cpu_gpu Float GetRadius(){ return radius; }
    bb_cpu_gpu Float GetMass(){ return mass; }
    bb_cpu_gpu int GetParticleCount(){ return count; }
    bb_cpu_gpu bool HasNormal(){ return normals.size > 0; }
    bb_cpu_gpu void SetFamilyId(unsigned int id){ familyId = id; }
    bb_cpu_gpu unsigned int GetFamilyId(){ return familyId; }

    bb_cpu_gpu Bucket *GetParticleBucket(int pId){
        AssertA(pId < count && pId >= 0, "Invalid set for particle bucket");
        return buckets.Get(pId);
    }

    bb_cpu_gpu ParticleChain *GetParticleAuxChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle aux chain node");
        return chainAuxNodes.Get(pId);
    }

    bb_cpu_gpu ParticleChain *GetParticleChainNode(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle chain node");
        return chainNodes.Get(pId);
    }

    bb_cpu_gpu Float GetParticleV0(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle v0");
        return v0s.At(pId);
    }

    bb_cpu_gpu Float GetParticleTemperature(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle temperature");
        return temperature.At(pId);
    }

    bb_cpu_gpu void SetParticleTemperature(int pId, Float temp){
        AssertA(pId < count && pId >= 0, "Invalid set for particle temperature");
        temperature.Set(temp, pId);
    }

    bb_cpu_gpu T GetParticleNormal(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle normal");
        return normals.At(pId);
    }

    bb_cpu_gpu Float GetParticleDensityEx(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle densityEx");
        return densitiesEx.At(pId);
    }

    bb_cpu_gpu T GetParticlePosition(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle position");
        return positions.At(pId);
    }

    bb_cpu_gpu T GetParticleVelocity(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle velocity");
        return velocities.At(pId);
    }

    bb_cpu_gpu T GetParticleForce(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle forces");
        return forces.At(pId);
    }

    bb_cpu_gpu Float GetParticleDensity(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle density");
        return densities.At(pId);
    }

    bb_cpu_gpu Float GetParticlePressure(int pId){
        AssertA(pId < count && pId >= 0, "Invalid query for particle pressure");
        return pressures.At(pId);
    }

    bb_cpu_gpu void SetParticleNormal(int pId, const T &nor){
        AssertA(pId < count && pId >= 0, "Invalid set for particle normal");
        normals.Set(nor, pId);
    }

    bb_cpu_gpu void SetParticlePosition(int pId, const T &pos){
        AssertA(pId < count && pId >= 0, "Invalid set for particle position");
        positions.Set(pos, pId);
    }
    bb_cpu_gpu void SetParticleVelocity(int pId, const T &vel){
        AssertA(pId < count && pId >= 0, "Invalid set for particle velocity");
        velocities.Set(vel, pId);
    }

    bb_cpu_gpu void SetParticleForce(int pId, const T &force){
        AssertA(pId < count && pId >= 0, "Invalid set for particle forces");
        forces.Set(force, pId);
    }

    bb_cpu_gpu void SetParticleDensity(int pId, Float density){
        AssertA(pId < count && pId >= 0, "Invalid set for particle density");
        densities.Set(density, pId);
    }

    bb_cpu_gpu void SetParticlePressure(int pId, Float pressure){
        AssertA(pId < count && pId >= 0, "Invalid set for particle pressure");
        pressures.Set(pressure, pId);
    }

    bb_cpu_gpu void SetParticleDensityEx(int pId, Float density){
        AssertA(pId < count && pId >= 0, "Invalid set for particle densityEx");
        densitiesEx.Set(density, pId);
    }

    bb_cpu_gpu void SetParticleV0(int pId, Float v0){
        AssertA(pId < count && pId >= 0, "Invalid set for particle v0s");
        v0s.Set(v0, pId);
    }

    bb_cpu_gpu vec3f GetParticleUserBufferVec3(int pId, int bufId){
        AssertA(pId < count && pId >= 0, "Invalid set for particle user vec3");
        AssertA(bufId >= 0 && bufId < userVec3Count, "Invalid buffer id for vec3 buffer");
        return userVec3Buffers[bufId].At(pId);
    }

    bb_cpu_gpu void SetParticleUserBufferVec3(int pId, vec3f value, int bufId){
        AssertA(pId < count && pId >= 0, "Invalid set for particle user vec3");
        AssertA(bufId >= 0 && bufId < userVec3Count, "Invalid buffer id for vec3 buffer");
        userVec3Buffers[bufId].Set(value, pId);
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

    bb_cpu_gpu SphParticleSet2() : targetDensity(WaterDensity), targetSpacing(0.1),
    kernelRadiusOverTargetSpacing(2.0)
    {
        particleSet = nullptr;
    }

    bb_cpu_gpu ParticleSet2 *GetParticleSet(){ return particleSet; }
    bb_cpu_gpu Float GetKernelRadius(){ return kernelRadius; }
    bb_cpu_gpu Float GetTargetDensity(){ return targetDensity; }
    bb_cpu_gpu Float GetTargetSpacing(){ return targetSpacing; }
    bb_cpu_gpu void SetHigherLevel(){ requiresHigherLevelUpdate = 1; }
    bb_cpu_gpu void ResetHigherLevel(){ requiresHigherLevelUpdate = 0; }

    bb_cpu_gpu void SetParticleData(ParticleSet2 *set){
        targetDensity = WaterDensity;
        targetSpacing = 0.1;
        kernelRadiusOverTargetSpacing = 2.0;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        particleSet = set;
    }

    bb_cpu_gpu void SetTargetSpacing(Float spacing){
        AssertA(particleSet, "Invalid call to SphParticleSet::SetTargetSpacing");
        particleSet->SetRadius(spacing);
        targetSpacing = spacing;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    bb_cpu_gpu void SetTargetDensity(Float density){
        targetDensity = density;
        ComputeMass();
    }

    bb_cpu_gpu void SetRelativeKernelRadius(Float relativeRadius){
        kernelRadiusOverTargetSpacing = relativeRadius;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    bb_cpu_gpu void ComputeMass(){
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

    bb_cpu_gpu unsigned int ComputeNumberOfTimeSteps(Float timeStep, Float speedOfSound,
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

    bb_cpu_gpu SphParticleSet3() : targetDensity(WaterDensity), targetSpacing(0.1),
    kernelRadiusOverTargetSpacing(2.0)
    {
        particleSet = nullptr;
    }

    bb_cpu_gpu ParticleSet3 *GetParticleSet(){ return particleSet; }
    bb_cpu_gpu Float GetKernelRadius(){ return kernelRadius; }
    bb_cpu_gpu Float GetTargetDensity(){ return targetDensity; }
    bb_cpu_gpu Float GetTargetSpacing(){ return targetSpacing; }
    bb_cpu_gpu void SetHigherLevel(){ requiresHigherLevelUpdate = 1; }
    bb_cpu_gpu void ResetHigherLevel(){ requiresHigherLevelUpdate = 0; }

    bb_cpu_gpu void SetParticleData(ParticleSet3 *set){
        targetDensity = WaterDensity;
        targetSpacing = 0.1;
        kernelRadiusOverTargetSpacing = 2.0;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        particleSet = set;
    }

    bb_cpu_gpu void SetTargetSpacing(Float spacing){
        AssertA(particleSet, "Invalid call to SphParticleSet::SetTargetSpacing");
        particleSet->SetRadius(spacing);
        targetSpacing = spacing;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    bb_cpu_gpu void SetTargetDensity(Float density){
        targetDensity = density;
        ComputeMass();
    }

    bb_cpu_gpu void SetRelativeKernelRadius(Float relativeRadius){
        kernelRadiusOverTargetSpacing = relativeRadius;
        kernelRadius = kernelRadiusOverTargetSpacing * targetSpacing;
        ComputeMass();
    }

    bb_cpu_gpu void ComputeMass(){
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

    bb_cpu_gpu unsigned int ComputeNumberOfTimeSteps(Float timeStep, Float speedOfSound,
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
    ParticleSetBuilder(){}

    void SetVelocityForAll(const T &vel){
        for(int i = 0; i < positions.size(); i++){
            velocities[i] = vel;
        }
    }

    int AddParticle(const T &pos, const T &vel = T(0), const T &force = T(0)){
        positions.push_back(pos);
        velocities.push_back(vel);
        forces.push_back(force);
        return 1;
    }

    int AddParticle(const T &pos, Float mpW, const T &vel = T(0)){
        positions.push_back(pos);
        velocities.push_back(vel);
        mpw.push_back(mpW);
        return 1;
    }

    void Commit(){}

    int GetParticleCount(){ return positions.size(); }

    SpecieSet<T> * MakeSpecieSet(Float mass, Float charge){
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

    ParticleSet<T> * MakeExtendedParticleSet(){
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

    ParticleSet<T> * MakeParticleSet(){
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

SphParticleSet2 *SphParticleSet2FromBuilder(ParticleSetBuilder2 *builder);
SphParticleSet2 *SphParticleSet2ExFromBuilder(ParticleSetBuilder2 *builder);
SpecieSet2 *SpecieSet2FromBuilder(ParticleSetBuilder2 *builder,
                                           Float mass, Float charge, int familyId = 0);

SphParticleSet3 *SphParticleSet3FromBuilder(ParticleSetBuilder3 *builder);
