#pragma once
#include <sph_solver.h>
#include <particle.h>
#include <vector>

#define SERIALIZER_POSITION  0x01
#define SERIALIZER_VELOCITY  0x02
#define SERIALIZER_DENSITY   0x04
#define SERIALIZER_BOUNDARY  0x08
#define SERIALIZER_NORMAL    0x10
#define SERIALIZER_MASS      0x20
#define SERIALIZER_LAYERS    0x40 // NOTE: This is not processed by serializer

/* Rules for writting */
#define SERIALIZER_RULE_BOUNDARY_EXCLUSIVE 0x100

/* For agnostic parsing use this structure */
struct SerializedParticle{
    vec3f position;
    vec3f velocity;
    vec3f normal;
    Float density;
    Float mass;
    int boundary;
};

void SerializerSaveSphDataSet3(SphSolverData3 *pSet, const char *filename, int flags,
                               std::vector<int> *boundary = nullptr);
void SerializerSaveSphDataSet2(SphSolverData2 *pSet, const char *filename, int flags,
                               std::vector<int> *boundary = nullptr);
int SerializerLoadSphDataSet3(ParticleSetBuilder3 *builder, const char *filename, 
                              int flags, std::vector<int> *boundary = nullptr);

void SerializerLoadPoints3(std::vector<vec3f> *points, const char *filename, int flags);

int SerializerLoadMany3(std::vector<vec3f> ***data, const char *basename, int flags,
                        int start, int end);

int SerializerLoadParticles3(std::vector<SerializedParticle> *pSet, 
                             const char *filename, int flags);

int SerializerFlagsFromString(const char *spec);

std::string SerializerStringFromFlags(int flags);
