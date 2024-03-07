#pragma once
#include <sph_solver.h>
#include <particle.h>
#include <vector>
#include <map>

#define SERIALIZER_POSITION  0x01
#define SERIALIZER_VELOCITY  0x02
#define SERIALIZER_DENSITY   0x04
#define SERIALIZER_BOUNDARY  0x08
#define SERIALIZER_NORMAL    0x10
#define SERIALIZER_MASS      0x20
#define SERIALIZER_LAYERS    0x40 // NOTE: This is not processed by serializer
#define SERIALIZER_XYZ       0x80 // NOTE: This flag overwrite other flags on legacy fmt

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

struct SerializedShape{
    ShapeType type;
    std::map<std::string, vec4f> numParameters;
    std::map<std::string, std::string> strParameters;
    std::map<std::string, Transform> transfParameters;
};

void SerializerSaveSphDataSet3(SphSolverData3 *pSet, const char *filename, int flags,
                               std::vector<int> *boundary = nullptr);

void SerializerSaveSphDataSet3Many(SphSolverData3 *data,
                                   std::vector<ParticleSet3 *> pSets,
                                   const char *filename, int flags);

void SerializerSaveSphDataSet2(SphSolverData2 *pSet, const char *filename, int flags,
                               std::vector<int> *boundary = nullptr);

void SerializerSaveSphDataSet3Legacy(SphSolverData3 *pSet, const char *filename,
                                     int flags, std::vector<int> *boundary = nullptr);

void SerializerSaveSphDataSet2Legacy(SphSolverData2 *pSet, const char *filename,
                                     int flags, std::vector<int> *boundary = nullptr);

void SerializerLoadLegacySystem3(std::vector<vec3f> *points, const char *filename,
                                 int flags, std::vector<int> *boundaries = nullptr);

int SerializerLoadSphDataSet3(ParticleSetBuilder3 *builder, const char *filename,
                              int &flags, std::vector<int> *boundary = nullptr);

void SerializerLoadPoints3(std::vector<vec3f> *points, const char *filename, int &flags);

int SerializerLoadMany3(std::vector<vec3f> ***data, const char *basename, int &flags,
                        int start, int end, int legacy=0);

int SerializerLoadParticles3(std::vector<SerializedParticle> *pSet,
                             const char *filename, int &flags);

void SerializerSaveDomain(SphSolverData3 *pSet, const char *filename);

void SerializerSaveDomain(SphSolverData2 *pSet, const char *filename);

void SerializerLoadSystem3(ParticleSetBuilder3 *builder,
                           std::vector<SerializedShape> *shapes,
                           const char *filename, int &flags,
                           std::vector<int> *boundary = nullptr);

void SerializerWriteShapes(std::vector<SerializedShape> *shapes, const char *filename);

int SerializerFlagsFromString(const char *spec);

std::string SerializerStringFromFlags(int flags);

const char *SerializerGetShapeName(ShapeType type);

void SerializerSetWrites(bool write);

bool SerializerIsWrittable();
