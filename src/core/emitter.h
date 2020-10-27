#pragma once
#include <geometry.h>
#include <shape.h>
#include <particle.h>

class VolumeParticleEmitter2{
    public:
    Shape2 *shape;
    Bounds2f bound;
    vec2f initVel, linearVel;
    Float angularVel, spacing, jitter;
    int maxParticles;
    int emittedParticles;
    bool isOneShot, allowOverlapping;
    PointGenerator2 *generator;
    
    __host__ VolumeParticleEmitter2(Shape2 *shape, const Bounds2f &bounds,
                                    Float spacing, const vec2f &initialVel = vec2f(0),
                                    const vec2f &linearVel = vec2f(0),
                                    Float angularVel = 0, int maxParticles = IntInfinity,
                                    Float jitter = 0, bool isOneShot = true,
                                    bool allowOverlapping = false, int seed = 0);
    
    __host__ void SetJitter(Float jitter);
    __host__ void Emit(ParticleSetBuilder<vec2f> *Builder);
};

class VolumeParticleEmitter3{
    public:
    Shape *shape;
    Bounds3f bound;
    vec3f initVel, linearVel;
    Float angularVel, spacing, jitter;
    int maxParticles;
    int emittedParticles;
    bool isOneShot, allowOverlapping;
    PointGenerator3 *generator;
    
    __host__ VolumeParticleEmitter3(Shape *shape, const Bounds3f &bounds,
                                    Float spacing, const vec3f &initialVel = vec3f(0),
                                    const vec3f &linearVel = vec3f(0),
                                    Float angularVel = 0, int maxParticles = IntInfinity,
                                    Float jitter = 0, bool isOneShot = true,
                                    bool allowOverlapping = false, int seed = 0);
    __host__ void SetJitter(Float jitter);
    __host__ void Emit(ParticleSetBuilder<vec3f> *Builder);
};

class UniformBoxParticleEmitter2{
    public:
    Bounds2f bound;
    vec2ui amount;
    __host__ UniformBoxParticleEmitter2(Bounds2f bound, vec2ui amount);
    __host__ void Emit(ParticleSetBuilder<vec2f> *Builder, Float number_density);
};

class VolumeParticleEmitterSet2{
    public:
    std::vector<VolumeParticleEmitter2 *> emitters;
    
    __host__ VolumeParticleEmitterSet2();
    __host__ void AddEmitter(VolumeParticleEmitter2 *emitter);
    __host__ void AddEmitter(Shape2 *shape, Float spacing, 
                             const vec2f &initialVel = vec2f(0),
                             const vec2f &linearVel = vec2f(0),
                             Float angularVel = 0, int maxParticles = IntInfinity,
                             Float jitter = 0, bool isOneShot = true,
                             bool allowOverlapping = false, int seed = 0);
    
    __host__ void AddEmitter(Shape2 *shape, const Bounds2f &bounds,
                             Float spacing, const vec2f &initialVel = vec2f(0),
                             const vec2f &linearVel = vec2f(0),
                             Float angularVel = 0, int maxParticles = IntInfinity,
                             Float jitter = 0, bool isOneShot = true,
                             bool allowOverlapping = false, int seed = 0);
    
    __host__ void SetJitter(Float jitter);
    __host__ void Emit(ParticleSetBuilder<vec2f> *Builder);
    __host__ void Release();
};

class VolumeParticleEmitterSet3{
    public:
    std::vector<VolumeParticleEmitter3 *> emitters;
    
    __host__ VolumeParticleEmitterSet3();
    __host__ void AddEmitter(VolumeParticleEmitter3 *emitter);
    __host__ void AddEmitter(Shape *shape, const Bounds3f &bounds,
                             Float spacing, const vec3f &initialVel = vec3f(0),
                             const vec3f &linearVel = vec3f(0),
                             Float angularVel = 0, int maxParticles = IntInfinity,
                             Float jitter = 0, bool isOneShot = true,
                             bool allowOverlapping = false, int seed = 0);
    
    __host__ void AddEmitter(Shape *shape, Float spacing, 
                             const vec3f &initialVel = vec3f(0),
                             const vec3f &linearVel = vec3f(0),
                             Float angularVel = 0, int maxParticles = IntInfinity,
                             Float jitter = 0, bool isOneShot = true,
                             bool allowOverlapping = false, int seed = 0);
    
    __host__ void SetJitter(Float jitter);
    __host__ void Emit(ParticleSetBuilder<vec3f> *Builder);
    __host__ void Release();
};