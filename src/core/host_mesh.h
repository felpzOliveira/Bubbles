/* date = October 13rd 2022 21:34 */
#pragma once
#include <geometry.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
/*
* Representation of a triangle mesh for surface reconstruction.
*/
typedef enum{
    FORMAT_OBJ,
    FORMAT_PLY,
    FORMAT_NONE,
}TriangleMeshFormat;

inline std::string FormatString(TriangleMeshFormat format){
    switch(format){
        case FORMAT_OBJ : return "OBJ";
        case FORMAT_PLY : return "PLY";
        default: return "UNKNOWN";
    }
}

inline TriangleMeshFormat FormatFromString(std::string str){
    if(str == "obj" || str == "OBJ") return FORMAT_OBJ;
    else if(str == "ply" || str == "PLY") return FORMAT_PLY;
    else return FORMAT_NONE;
}

class HostTriangleMesh3;
void writeObj(HostTriangleMesh3 *mesh, std::ostream *strm);
void writePly(HostTriangleMesh3 *mesh, std::ostream *strm);

class HostTriangleMesh3{
    public:
    std::vector<vec3f> points;
    std::vector<vec3f> normals;
    std::vector<vec2f> uvs;
    std::vector<vec3ui> pointIndices;
    std::vector<vec3ui> normalIndices;
    std::vector<vec3ui> uvIndices;

    HostTriangleMesh3(){}

    size_t numberOfPoints(){ return points.size(); }
    size_t numberOfTriangles(){ return pointIndices.size(); }

    bool hasUvs(){ return uvs.size() > 0; }
    bool hasNormals(){ return normals.size() > 0; }

    void addPoint(vec3f p){ points.push_back(p); }
    void addNormal(vec3f n){ normals.push_back(n); }
    void addUv(vec2f uv){ uvs.push_back(uv); }
    void addPointUvNormalTriangle(vec3ui np, vec3ui nuv, vec3ui nno){
        pointIndices.push_back(np);
        uvIndices.push_back(nuv);
        normalIndices.push_back(nno);
    }

    void writeToDisk(const char *path, TriangleMeshFormat format){
        std::ofstream file(path);
        if(file){
            switch(format){
                case FORMAT_OBJ:{
                    writeObj(this, &file);
                } break;
                case FORMAT_PLY:{
                    writePly(this, &file);
                } break;
                default:{
                    printf("Unknown format\n");
                }
            }
            file.close();
        }else{
            printf("Cannot open file %s\n", path);
        }
    }
};
