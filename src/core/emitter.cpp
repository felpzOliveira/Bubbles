#include <emitter.h>
#include <statics.h>

__host__ vec2f ZeroVelocityField2(const vec2f &p){ return vec2f(0); }
__host__ vec3f ZeroVelocityField3(const vec3f &p){ return vec3f(0); }

__bidevice__ vec3f SampleSphere(const vec2f &u){
    Float usqrt = 2 * std::sqrt(u[1] * (1 - u[1]));
    Float utheta = 2 * Pi * u[0];
    return vec3f(std::cos(utheta) * usqrt, std::sin(utheta) * usqrt, 1 - 2*u[1]);
}


/**************************************************************/
//               U N I F O R M   B O X   2 D                  //
/**************************************************************/
__host__ UniformBoxParticleEmitter2::UniformBoxParticleEmitter2(Bounds2f bound,
                                                                vec2ui amount)
:bound(bound), amount(amount){}

__host__ void UniformBoxParticleEmitter2::Emit(ParticleSetBuilder<vec2f> *Builder,
                                               Float number_density)
{
    Float area  = bound.SurfaceArea();
    int total   = (amount.x - 1) * (amount.y - 1);
    Float nReal = number_density * area;
    Float mpw   = nReal / (Float)total;

    Float hx = bound.ExtentOn(0) / (amount.x - 1);
    Float hy = bound.ExtentOn(1) / (amount.y - 1);

    vec2f p0 = bound.pMin;
    vec2f p1 = bound.pMax;
    for(int i = 0; i < amount.x; i++){
        for(int j = 0; j < amount.y; j++){
            vec2f target = p0 + vec2f(i * hx, j * hy);
            if(target.x == p1.x) target.x -= 1e-4 * hx;
            if(target.y == p1.y) target.y -= 1e-4 * hy;

            Float w = 1.0;
            if(i == 0 || i == amount.x-1) w *= 0.5;
            if(j == 0 || j == amount.y-1) w *= 0.5;

            if(!Builder->AddParticle(target, w * mpw)){
                return;
            }
        }
    }
}

/**************************************************************/
//          V O L U M E    P A R T I C L E    2 D             //
/**************************************************************/
__host__ VolumeParticleEmitter2::VolumeParticleEmitter2(Shape2 *shape, const Bounds2f &bounds,
                                                        Float spacing, const vec2f &initialVel,
                                                        const vec2f &linearVel,
                                                        Float angularVel,
                                                        int maxParticles, Float jitter,
                                                        bool isOneShot,bool allowOverlapping,
                                                        int seed)
: shape(shape), bound(bounds), spacing(spacing), initVel(initialVel), linearVel(linearVel),
angularVel(angularVel), maxParticles(maxParticles), emittedParticles(0), jitter(jitter),
isOneShot(isOneShot), allowOverlapping(allowOverlapping)
{
    (void)seed;
    generator = new TrianglePointGenerator();
    emittedParticles = 0;
}

__host__ void VolumeParticleEmitter2::SetJitter(Float jit){
    jitter = Clamp(jit, 0.0, 1.0);
}

template<typename ParticleBuilder2>
__host__ void _Emit(ParticleBuilder2 *Builder, VolumeParticleEmitter2 *emitter,
                    const std::function<vec2f(const vec2f &)> &velocity)
{
    AssertA(emitter->shape, "Cannot emit particles without a surface shape");

    Float maxJitter = 0.5 * emitter->jitter * emitter->spacing;
    int numNewParticles = 0;

    AssertA(emitter->isOneShot, "Multiple emittion not supported yet");

    if(emitter->isOneShot){
        auto Accept = [&](const vec2f &point) -> bool{
            Float angle = (rand_float() - 0.5) * 2.0 * Pi;
            vec2f randomDir = vec2f(1, 1).Rotate(angle);
            vec2f offset = maxJitter * randomDir;
            vec2f target = point + offset;
            if(emitter->shape->SignedDistance(target) <= 0){
                if(emitter->emittedParticles < emitter->maxParticles){
                    vec2f vel = velocity(target) + emitter->initVel;
                    if(Builder->AddParticle(target, vel)){
                        emitter->emittedParticles++;
                        numNewParticles++;
                    }else{
                        return false;
                    }
                }else{
                    return false;
                }
            }

            return true;
        };

        emitter->generator->ForEach(emitter->bound, emitter->spacing, Accept);
        DBG_PRINT("Added %d particles\n", numNewParticles);
        if(numNewParticles > 0) Builder->Commit();
    }
}

__host__ void VolumeParticleEmitter2::Emit(ContinuousParticleSetBuilder2 *Builder,
                                           const std::function<vec2f(const vec2f &)> &velocity)
{
    _Emit<ContinuousParticleSetBuilder2>(Builder, this, velocity);
}

__host__ void VolumeParticleEmitter2::Emit(ParticleSetBuilder<vec2f> *Builder,
                                           const std::function<vec2f(const vec2f &)> &velocity)
{
    _Emit<ParticleSetBuilder<vec2f>>(Builder, this, velocity);
}

/**************************************************************/
//          V O L U M E    P A R T I C L E    S E T   2 D     //
/**************************************************************/
__host__ VolumeParticleEmitterSet2::VolumeParticleEmitterSet2(){}

__host__ void VolumeParticleEmitterSet2::AddEmitter(VolumeParticleEmitter2 *emitter){
    emitters.push_back(emitter);
}

__host__ void VolumeParticleEmitterSet2::AddEmitter(Shape2 *shape, const Bounds2f &bounds,
                                                    Float spacing, const vec2f &initialVel,
                                                    const vec2f &linearVel,
                                                    Float angularVel, int maxParticles,
                                                    Float jitter, bool isOneShot,
                                                    bool allowOverlapping, int seed)
{
    emitters.push_back(new VolumeParticleEmitter2(shape, bounds, spacing, initialVel,
                                                  linearVel, angularVel, maxParticles,
                                                  jitter, isOneShot, allowOverlapping, seed));
}

__host__ void VolumeParticleEmitterSet2::AddEmitter(Shape2 *shape, Float spacing,
                                                    const vec2f &initialVel,
                                                    const vec2f &linearVel,
                                                    Float angularVel, int maxParticles,
                                                    Float jitter, bool isOneShot,
                                                    bool allowOverlapping, int seed)
{
    emitters.push_back(new VolumeParticleEmitter2(shape, shape->GetBounds(), spacing,
                                                  initialVel,linearVel, angularVel,
                                                  maxParticles, jitter, isOneShot,
                                                  allowOverlapping, seed));
}

__host__ void VolumeParticleEmitterSet2::SetJitter(Float jitter){
    AssertA(emitters.size() > 0, "No Emitter given for VolumeParticleEmitterSet2::SetJitter");
    for(int i = 0; i < emitters.size(); i++){
        emitters[i]->SetJitter(jitter);
    }
}

__host__ void VolumeParticleEmitterSet2::Emit(ContinuousParticleSetBuilder2 *Builder,
                                              const std::function<vec2f(const vec2f &)> &velocity)
{
    AssertA(emitters.size() > 0, "No Emitter given for VolumeParticleEmitterSet2::Emit");
    for(int i = 0; i < emitters.size(); i++){
        emitters[i]->Emit(Builder, velocity);
    }
}

__host__ void VolumeParticleEmitterSet2::Emit(ParticleSetBuilder<vec2f> *Builder,
                                              const std::function<vec2f(const vec2f &)> &velocity){
    AssertA(emitters.size() > 0, "No Emitter given for VolumeParticleEmitterSet2::Emit");
    for(int i = 0; i < emitters.size(); i++){
        emitters[i]->Emit(Builder, velocity);
    }
}

__host__ void VolumeParticleEmitterSet2::Release(){
    for(int i = 0; i < emitters.size(); i++){
        if(emitters[i]){
            delete emitters[i];
        }
    }

    emitters.clear();
}

/**************************************************************/
//          V O L U M E    P A R T I C L E    3 D             //
/**************************************************************/
__host__ VolumeParticleEmitter3::VolumeParticleEmitter3(Shape *shape, const Bounds3f &bounds,
                                                        Float spacing, const vec3f &initialVel,
                                                        const vec3f &linearVel,
                                                        Float angularVel, int maxParticles,
                                                        Float jitter, bool isOneShot,
                                                        bool allowOverlapping, int seed)
: shape(shape), bound(bounds), spacing(spacing), initVel(initialVel), linearVel(linearVel),
angularVel(angularVel), maxParticles(maxParticles), emittedParticles(0), jitter(jitter),
isOneShot(isOneShot), allowOverlapping(allowOverlapping)
{
    (void)seed;
    generator = new BccLatticePointGenerator();
    withValidator = 0;
    emittedParticles = 0;
}
__host__ VolumeParticleEmitter3::VolumeParticleEmitter3(Shape *_shape, Float spacing,
                                                        const vec3f &initialVel,
                                                        const vec3f &linearVel,
                                                        Float angularVel, int maxParticles,
                                                        Float jitter, bool isOneShot,
                                                        bool allowOverlapping, int seed)
: shape(_shape), bound(_shape->GetBounds()), spacing(spacing), initVel(initialVel), linearVel(linearVel),
angularVel(angularVel), maxParticles(maxParticles), emittedParticles(0), jitter(jitter),
isOneShot(isOneShot), allowOverlapping(allowOverlapping)
{
    (void)seed;
    generator = new BccLatticePointGenerator();
    emittedParticles = 0;
    withValidator = 0;
}

__host__ void
VolumeParticleEmitter3::SetValidator(std::function<int(const vec3f &)> gValidator){
    withValidator = 1;
    validatorFunc = gValidator;
}


__host__ void VolumeParticleEmitter3::SetJitter(Float jit){
    jitter = Clamp(jit, 0.0, 1.0);
}

template<typename ParticleBuilder3>
__host__ void _Emit(ParticleBuilder3 *Builder, VolumeParticleEmitter3 *emitter,
                    const std::function<vec3f(const vec3f &)> &velocity)
{
    AssertA(emitter->isOneShot, "Multiple emittion not supported yet");
    TimerList timers;
    Float maxJitter = 0.5 * emitter->jitter * emitter->spacing;
    int numNewParticles = 0;
    bool isSdf = emitter->shape->CanSolveSdf();

    if(emitter->isOneShot){
        auto Accept = [&](const vec3f &point) -> bool{
            if(emitter->withValidator){
                if(!emitter->validatorFunc(point)) return true;
            }

            vec2f u(rand_float(), rand_float());
            vec3f randomDir = SampleSphere(u);
            vec3f offset = maxJitter * randomDir;
            vec3f target = point + offset;
            if(isSdf){
                if(emitter->shape->SignedDistance(target) <= 0){
                    if(emitter->emittedParticles < emitter->maxParticles){
                        vec3f vel = velocity(target) + emitter->initVel;
                        if(Builder->AddParticle(target, vel)){
                            emitter->emittedParticles++;
                            numNewParticles++;
                        }else{
                            return false;
                        }
                    }else{
                        return false;
                    }
                }
            }else{
                /*
                * If the Shape does not provide a SDF map, perform Ray Tracing
                * for detecting interior points, for meshes only.
                */
                if(MeshIsPointInside(target, emitter->shape, emitter->bound)){
                    if(emitter->emittedParticles < emitter->maxParticles){
                        vec3f vel = velocity(target) + emitter->initVel;
                        if(Builder->AddParticle(target, vel)){
                            emitter->emittedParticles++;
                            numNewParticles++;
                        }else{
                            return false;
                        }
                    }else{
                        return false;
                    }
                }
            }

            return true;
        };

        printf("Emitting particles, spacing = %g ... ", emitter->spacing);
        fflush(stdout);
        timers.Start();
        emitter->generator->ForEach(emitter->bound, emitter->spacing, Accept);
        timers.Stop();
        printf(" %d { %g ms }\n", numNewParticles, timers.GetElapsedCPU(0));
        if(numNewParticles > 0) Builder->Commit();
    }
}

__host__ void VolumeParticleEmitter3::Emit(ContinuousParticleSetBuilder3 *Builder,
                                           const std::function<vec3f(const vec3f &)> &velocity)
{
    _Emit<ContinuousParticleSetBuilder3>(Builder, this, velocity);
}

__host__ void VolumeParticleEmitter3::Emit(ParticleSetBuilder<vec3f> *Builder,
                                           const std::function<vec3f(const vec3f &)> &velocity)
{
    _Emit<ParticleSetBuilder<vec3f>>(Builder, this, velocity);
}

/**************************************************************/
//          V O L U M E    P A R T I C L E    S E T   3 D     //
/**************************************************************/
__host__ VolumeParticleEmitterSet3::VolumeParticleEmitterSet3(){}

__host__ void VolumeParticleEmitterSet3::AddEmitter(VolumeParticleEmitter3 *emitter){
    emitters.push_back(emitter);
}

__host__ void VolumeParticleEmitterSet3::AddEmitter(Shape *shape, Float spacing,
                                                    const vec3f &initialVel,
                                                    const vec3f &linearVel,
                                                    Float angularVel, int maxParticles,
                                                    Float jitter, bool isOneShot,
                                                    bool allowOverlapping, int seed)
{
    emitters.push_back(new VolumeParticleEmitter3(shape, shape->GetBounds(), spacing,
                                                  initialVel, linearVel, angularVel,
                                                  maxParticles, jitter, isOneShot,
                                                  allowOverlapping, seed));
}

__host__ void VolumeParticleEmitterSet3::AddEmitter(Shape *shape, const Bounds3f &bounds,
                                                    Float spacing, const vec3f &initialVel,
                                                    const vec3f &linearVel,
                                                    Float angularVel, int maxParticles,
                                                    Float jitter, bool isOneShot,
                                                    bool allowOverlapping, int seed)
{
    emitters.push_back(new VolumeParticleEmitter3(shape, bounds, spacing, initialVel,
                                                  linearVel, angularVel, maxParticles,
                                                  jitter, isOneShot, allowOverlapping, seed));
}

__host__ void
VolumeParticleEmitterSet3::SetValidator(std::function<int(const vec3f &)> gValidator){
    for(VolumeParticleEmitter3 *emitter : emitters){
        emitter->SetValidator(gValidator);
    }
}

__host__ void VolumeParticleEmitterSet3::SetJitter(Float jitter){
    AssertA(emitters.size() > 0, "No Emitter given for VolumeParticleEmitterSet3::SetJitter");
    for(int i = 0; i < emitters.size(); i++){
        emitters[i]->SetJitter(jitter);
    }
}

__host__ void VolumeParticleEmitterSet3::Emit(ContinuousParticleSetBuilder3 *Builder,
                                              const std::function<vec3f(const vec3f &)> &velocity)
{
    AssertA(emitters.size() > 0, "No Emitter given for VolumeParticleEmitterSet3::Emit");
    for(int i = 0; i < emitters.size(); i++){
        emitters[i]->Emit(Builder, velocity);
    }
}

__host__ void VolumeParticleEmitterSet3::Emit(ParticleSetBuilder<vec3f> *Builder,
                                              const std::function<vec3f(const vec3f &)> &velocity)
{
    AssertA(emitters.size() > 0, "No Emitter given for VolumeParticleEmitterSet3::Emit");
    for(int i = 0; i < emitters.size(); i++){
        emitters[i]->Emit(Builder, velocity);
    }
}

__host__ void VolumeParticleEmitterSet3::Release(){
    for(int i = 0; i < emitters.size(); i++){
        if(emitters[i]){
            delete emitters[i];
        }
    }

    emitters.clear();
}
