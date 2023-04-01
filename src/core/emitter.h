#pragma once
#include <geometry.h>
#include <shape.h>
#include <particle.h>
#include <functional>

__host__ vec2f ZeroVelocityField2(const vec2f &p);
__host__ vec3f ZeroVelocityField3(const vec3f &p);

//TODO: Add validator to other emitters not only VolumeParticleEmitter3
//TODO: Make the VolumeEmitterSet* receive many velocity fields for each emitter
//      or at least return an id for the emitter being queried.

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
    __host__ void Emit(ParticleSetBuilder<vec2f> *Builder,
                       const std::function<vec2f(const vec2f &)> &velocity=ZeroVelocityField2);
    __host__ void Emit(ContinuousParticleSetBuilder2 *Builder,
                       const std::function<vec2f(const vec2f &)> &velocity=ZeroVelocityField2);
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
    std::function<int(const vec3f &)> validatorFunc;
    int withValidator;

    __host__ VolumeParticleEmitter3(Shape *shape, const Bounds3f &bounds,
                                    Float spacing, const vec3f &initialVel = vec3f(0),
                                    const vec3f &linearVel = vec3f(0),
                                    Float angularVel = 0, int maxParticles = IntInfinity,
                                    Float jitter = 0, bool isOneShot = true,
                                    bool allowOverlapping = false, int seed = 0);
    __host__ VolumeParticleEmitter3(Shape *shape, Float spacing,
                                    const vec3f &initialVel = vec3f(0),
                                    const vec3f &linearVel = vec3f(0),
                                    Float angularVel = 0, int maxParticles = IntInfinity,
                                    Float jitter = 0, bool isOneShot = true,
                                    bool allowOverlapping = false, int seed = 0);
    __host__ void SetValidator(std::function<int(const vec3f &)> gValidator);
    __host__ void SetJitter(Float jitter);
    __host__ void Emit(ParticleSetBuilder<vec3f> *Builder,
                       const std::function<vec3f(const vec3f &)> &velocity=ZeroVelocityField3);
    __host__ void Emit(ContinuousParticleSetBuilder3 *Builder,
                       const std::function<vec3f(const vec3f &)> &velocity=ZeroVelocityField3);
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
    __host__ void Emit(ParticleSetBuilder<vec2f> *Builder,
                       const std::function<vec2f(const vec2f &)> &velocity=ZeroVelocityField2);
    __host__ void Emit(ContinuousParticleSetBuilder2 *Builder,
                       const std::function<vec2f(const vec2f &)> &velocity=ZeroVelocityField2);
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

    __host__ void SetValidator(std::function<int(const vec3f &)> gValidator);
    __host__ void SetJitter(Float jitter);
    __host__ void Emit(ParticleSetBuilder<vec3f> *Builder,
                       const std::function<vec3f(const vec3f &)> &velocity=ZeroVelocityField3);
    __host__ void Emit(ContinuousParticleSetBuilder3 *Builder,
                       const std::function<vec3f(const vec3f &)> &velocity=ZeroVelocityField3);
    __host__ void Release();
};
