#include <marching_cubes_table.h>
#include <geometry.h>
#include <grid.h>
#include <obj_loader.h>
#include <unordered_map>
#include <marching_cubes.h>

/*
 * This implementation is from Jet. I'm re-implementing because jet is not compilable
 * under cuda. I'm unable to make nvcc accept their template parametrization.
 */

typedef size_t MarchingCubeVertexHashKey;
typedef size_t MarchingCubeVertexId;

typedef std::unordered_map<MarchingCubeVertexHashKey,
                           MarchingCubeVertexId> MarchingCubeVertexMap;

template <typename T>
T distanceToZeroLevelSet(T phi0, T phi1) {
    if(std::fabs(phi0) + std::fabs(phi1) > Epsilon){
        return std::fabs(phi0) / (std::fabs(phi0) + std::fabs(phi1));
    }else{
        return static_cast<T>(0.5);
    }
}

inline bool queryVertexId(const MarchingCubeVertexMap& vertexMap,
                          MarchingCubeVertexHashKey vKey, MarchingCubeVertexId* vId)
{
    auto vItr = vertexMap.find(vKey);
    if(vItr != vertexMap.end()){
        *vId = vItr->second;
        return true;
    }else{
        return false;
    }
}

inline vec3f grad(FieldGrid3f *grid, size_t i, size_t j, size_t k, const vec3f &invSize)
{
    vec3f ret;
    ssize_t ip = i + 1;
    ssize_t im = i - 1;
    ssize_t jp = j + 1;
    ssize_t jm = j - 1;
    ssize_t kp = k + 1;
    ssize_t km = k - 1;
    vec3ui dim = grid->GetResolution();
    ssize_t dimx = static_cast<ssize_t>(dim.x);
    ssize_t dimy = static_cast<ssize_t>(dim.y);
    ssize_t dimz = static_cast<ssize_t>(dim.z);
    if (i > dimx - 2) {
        ip = i;
    } else if (i == 0) {
        im = 0;
    }
    if (j > dimy - 2) {
        jp = j;
    } else if (j == 0) {
        jm = 0;
    }
    if (k > dimz - 2) {
        kp = k;
    } else if (k == 0) {
        km = 0;
    }
    ret.x = 0.5f * invSize.x * (grid->GetValueAt(ip, j, k) - grid->GetValueAt(im, j, k));
    ret.y = 0.5f * invSize.y * (grid->GetValueAt(i, jp, k) - grid->GetValueAt(i, jm, k));
    ret.z = 0.5f * invSize.z * (grid->GetValueAt(i, j, kp) - grid->GetValueAt(i, j, km));
    return ret;
}

inline vec3f safeNormalize(const vec3f& n) {
    if(n.LengthSquared() > 0.0){
        return Normalize(n);
    }else{
        return n;
    }
}

// To compute unique edge ID, map vertices+edges into
// doubled virtual vertex indices.
//
// v  edge   v
// |----*----|    -->    |-----|-----|
// i        i+1         2i   2i+1  2i+2
//
inline size_t globalEdgeId(size_t i, size_t j, size_t k, const vec3ui& dim,
                           size_t localEdgeId)
{
    // See edgeConnection in marching_cubes_table.h for the edge ordering.
    static const int edgeOffset3D[12][3] = {
        {1, 0, 0}, {2, 0, 1}, {1, 0, 2}, {0, 0, 1}, {1, 2, 0}, {2, 2, 1},
        {1, 2, 2}, {0, 2, 1}, {0, 1, 0}, {2, 1, 0}, {2, 1, 2}, {0, 1, 2}};

    return ((2 * k + edgeOffset3D[localEdgeId][2]) * 2 * dim.y +
            (2 * j + edgeOffset3D[localEdgeId][1])) * 2 * dim.x +
            (2 * i + edgeOffset3D[localEdgeId][0]);
}

// To compute unique edge ID, map vertices+edges into
// doubled virtual vertex indices.
//
// v  edge   v
// |----*----|    -->    |-----|-----|
// i        i+1         2i   2i+1  2i+2
//
inline size_t globalVertexId(size_t i, size_t j, size_t k, const vec3ui& dim,
                             size_t localVertexId)
{
    // See edgeConnection in marching_cubes_table.h for the edge ordering.
    static const int vertexOffset3D[8][3] = {{0, 0, 0}, {2, 0, 0}, {2, 0, 2},
                                             {0, 0, 2}, {0, 2, 0}, {2, 2, 0},
                                             {2, 2, 2}, {0, 2, 2}};

    return ((2 * k + vertexOffset3D[localVertexId][2]) * 2 * dim.y +
            (2 * j + vertexOffset3D[localVertexId][1])) * 2 * dim.x +
            (2 * i + vertexOffset3D[localVertexId][0]);
}

static void singleCube(const std::array<Float, 8> &data,
                       const std::array<size_t, 12> &edgeIds,
                       const std::array<vec3f, 8> &normals,
                       Bounds3f bound,
                       MarchingCubeVertexMap *vertexMap, HostTriangleMesh3 *mesh,
                       Float isoValue, bool rotate_face)
{
    int itrVertex, itrEdge, itrTri;
    int idxFlagSize = 0, idxEdgeFlags = 0;
    int idxVertexOfTheEdge[2];

    vec3f pos, pos0, pos1, normal, normal0, normal1;
    Float phi0, phi1;
    Float alpha;
    vec3f e[12], n[12];

    // Which vertices are inside? If i-th vertex is inside, mark '1' at i-th
    // bit. of 'idxFlagSize'.
    for (itrVertex = 0; itrVertex < 8; itrVertex++) {
        if (data[itrVertex] <= isoValue) {
            idxFlagSize |= 1 << itrVertex;
        }
    }

    // If the cube is entirely inside or outside of the surface, there is no job
    // to be done in this marching-cube cell.
    if (idxFlagSize == 0 || idxFlagSize == 255) {
        return;
    }

    // If there are vertices which is inside the surface...
    // Which edges intersect the surface? If i-th edge intersects the surface,
    // mark '1' at i-th bit of 'itrEdgeFlags'
    idxEdgeFlags = cubeEdgeFlags[idxFlagSize];

    // Find the point of intersection of the surface with each edge
    for (itrEdge = 0; itrEdge < 12; itrEdge++) {
        // If there is an intersection on this edge
        if (idxEdgeFlags & (1 << itrEdge)) {
            idxVertexOfTheEdge[0] = edgeConnection[itrEdge][0];
            idxVertexOfTheEdge[1] = edgeConnection[itrEdge][1];

            // cube vertex ordering to x-major ordering
            static int indexMap[8] = {0, 1, 5, 4, 2, 3, 7, 6};

            // Find the phi = 0 position
            pos0 = bound.Corner(indexMap[idxVertexOfTheEdge[0]]);
            pos1 = bound.Corner(indexMap[idxVertexOfTheEdge[1]]);

            normal0 = normals[idxVertexOfTheEdge[0]];
            normal1 = normals[idxVertexOfTheEdge[1]];

            phi0 = data[idxVertexOfTheEdge[0]] - isoValue;
            phi1 = data[idxVertexOfTheEdge[1]] - isoValue;

            alpha = distanceToZeroLevelSet(phi0, phi1);

            if(alpha < 0.000001){
                alpha = 0.000001;
            }
            if(alpha > 0.999999){
                alpha = 0.999999;
            }

            pos = Lerp(pos0, pos1, alpha);
            normal = Lerp(normal0, normal1, alpha);

            //pos = (1.0 - alpha) * pos0 + alpha * pos1;
            //normal = (1.0 - alpha) * normal0 + alpha * normal1;

            e[itrEdge] = pos;
            n[itrEdge] = normal;
        }
    }

    // Make triangles
    for (itrTri = 0; itrTri < 5; ++itrTri) {
        // If there isn't any triangle to be made, escape this loop.
        if (triangleConnectionTable3D[idxFlagSize][3 * itrTri] < 0) {
            break;
        }

        vec3ui face;

        for (int j = 0; j < 3; j++) {
            int k = 3 * itrTri + j;
            MarchingCubeVertexHashKey vKey =
                edgeIds[triangleConnectionTable3D[idxFlagSize][k]];
            MarchingCubeVertexId vId;
            if (queryVertexId(*vertexMap, vKey, &vId)) {
                face[j] = vId;
            } else {
                // If vertex does not exist from the map
                face[j] = mesh->numberOfPoints();
                mesh->addNormal(safeNormalize(n[triangleConnectionTable3D[idxFlagSize][k]]));
                mesh->addPoint(e[triangleConnectionTable3D[idxFlagSize][k]]);
                mesh->addUv(vec2f());
                vertexMap->insert(std::make_pair(vKey, face[j]));
            }
        }
        vec3ui actualFace = face;
        if(rotate_face){
            actualFace[0] = face[0];
            actualFace[1] = face[2];
            actualFace[2] = face[1];
        }
        mesh->addPointUvNormalTriangle(actualFace, actualFace, actualFace);
    }
}

void MarchingCubes(FieldGrid3f *grid, HostTriangleMesh3 *mesh, Float isoValue,
                   std::function<void(vec3ui u)> fn, bool rotate_faces)
{
    MarchingCubeVertexMap vertexMap;
    vec3f gridSize = grid->GetSpacing();

    const vec3ui dim = grid->GetResolution();
    const vec3f invGridSize(1.0 / (Float)gridSize.x,
                            1.0 / (Float)gridSize.y,
                            1.0 / (Float)gridSize.z);

    vec3ui inf(dim.x + 1, dim.y + 1, dim.z + 1);

    auto pos = [&](ssize_t i, ssize_t j, ssize_t k){
        return grid->GetDataPosition(vec3ui(i, j, k));
    };

    ssize_t dimx = static_cast<ssize_t>(dim.x);
    ssize_t dimy = static_cast<ssize_t>(dim.y);
    ssize_t dimz = static_cast<ssize_t>(dim.z);

    for (ssize_t k = 0; k < dimz - 1; ++k) {
        for (ssize_t j = 0; j < dimy - 1; ++j) {
            for (ssize_t i = 0; i < dimx - 1; ++i) {
                std::array<Float, 8> data;
                std::array<size_t, 12> edgeIds;
                std::array<vec3f, 8> normals;

                data[0] = grid->GetValueAt(i, j, k);
                data[1] = grid->GetValueAt(i + 1, j, k);
                data[4] = grid->GetValueAt(i, j + 1, k);
                data[5] = grid->GetValueAt(i + 1, j + 1, k);
                data[3] = grid->GetValueAt(i, j, k + 1);
                data[2] = grid->GetValueAt(i + 1, j, k + 1);
                data[7] = grid->GetValueAt(i, j + 1, k + 1);
                data[6] = grid->GetValueAt(i + 1, j + 1, k + 1);

                normals[0] = grad(grid, i, j, k, invGridSize);
                normals[1] = grad(grid, i + 1, j, k, invGridSize);
                normals[4] = grad(grid, i, j + 1, k, invGridSize);
                normals[5] = grad(grid, i + 1, j + 1, k, invGridSize);
                normals[3] = grad(grid, i, j, k + 1, invGridSize);
                normals[2] = grad(grid, i + 1, j, k + 1, invGridSize);
                normals[7] = grad(grid, i, j + 1, k + 1, invGridSize);
                normals[6] = grad(grid, i + 1, j + 1, k + 1, invGridSize);

                for (int e = 0; e < 12; e++) {
                    edgeIds[e] = globalEdgeId(i, j, k, dim, e);
                }

                Bounds3f bound(pos(i, j, k), pos(i + 1, j + 1, k + 1));

                singleCube(data, edgeIds, normals, bound, &vertexMap,
                            mesh, isoValue, rotate_faces);
                fn(vec3ui(i, j, k));
            }  // i
        }      // j
    }          // k

    fn(inf);
}
