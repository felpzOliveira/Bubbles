#pragma once
#include <vector>
#include <geometry.h>
#include <transform.h>
#include <grid.h>
#include <emitter.h>
#include <collider.h>
#include <graphy.h>
#include <functional>

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
__host__ Transform UtilComputeFitTransform(ParsedMesh *mesh, Float maximumAxisLength);

/*
* Generates a acceleration Grid3 for a domain given its bounds, the target spacing
* of the simulation and the spacing scale to be used.
*/
__host__ Grid3 *UtilBuildGridForDomain(Bounds3f domain, Float spacing, 
                                       Float spacingScale = 1.8);

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
                                                Transform transform=Transform(),
                                                vec3f centerAt=vec3f(0),
                                                vec3f initialVelocity=vec3f(0));

/*
* Run a simulation. Perform several updates on the given solver and display results
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
        UtilGetSDFParticles(shape->grid, &particles, 0, spacing);
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
        printf("Step: %d            \n", frame+1);
        frame++;
    }
    
    graphy_close_display();
    delete[] ptr;
}
