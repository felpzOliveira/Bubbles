#include <marching_cubes_table.h>
#include <geometry.h>
#include <grid.h>
#include <obj_loader.h>
#include <unordered_map>
#include <marching_cubes.h>
#include <cutil.h>

/*
* The base implementation is taken from the Jet framework, I adapted it
* to be runnable under CUDA.
*/

typedef size_t MarchingCubeVertexHashKey;
typedef size_t MarchingCubeVertexId;

struct MarchingCubeSingleCubeOutput{
    MarchingCubeVertexHashKey vKey[3];
    int k[3];
    size_t cubeIndex;
    bool valid;
};

struct MarchingCubeWorkItem{
    size_t cubeIndex;
    int idxFlagSize;
    Bounds3f bounds;
    Float data[8];
    vec3f normal[8];
    size_t edgeIds[12];
    vec3f e[12];
    vec3f n[12];

    bb_cpu_gpu
    MarchingCubeWorkItem(int){
        cubeIndex = 0;
        idxFlagSize = 0;
    }

    bb_cpu_gpu
    MarchingCubeWorkItem(size_t idx, int flag, Bounds3f _bounds,
                         Float *_data, vec3f *_normal, size_t *_edges)
    {
        cubeIndex = idx;
        idxFlagSize = flag;
        bounds = _bounds;
        data[0]= _data[0]; edgeIds[0] = _edges[0]; normal[0] = _normal[0];
        data[1]= _data[1]; edgeIds[1] = _edges[1]; normal[1] = _normal[1];
        data[2]= _data[2]; edgeIds[2] = _edges[2]; normal[2] = _normal[2];
        data[3]= _data[3]; edgeIds[3] = _edges[3]; normal[3] = _normal[3];
        data[4]= _data[4]; edgeIds[4] = _edges[4]; normal[4] = _normal[4];
        data[5]= _data[5]; edgeIds[5] = _edges[5]; normal[5] = _normal[5];
        data[6]= _data[6]; edgeIds[6] = _edges[6]; normal[6] = _normal[6];
        data[7]= _data[7]; edgeIds[7] = _edges[7]; normal[7] = _normal[7];
        edgeIds[8] = _edges[8];
        edgeIds[9] = _edges[9];
        edgeIds[10] = _edges[10];
        edgeIds[11] = _edges[11];
    }
};

typedef WorkQueue<MarchingCubeWorkItem> MarchingCubeWorkQueue;

typedef std::unordered_map<MarchingCubeVertexHashKey,
                           MarchingCubeVertexId> MarchingCubeVertexMap;

bb_gpu __constant__ int gpuEdgeConnection[12][2];
bb_gpu __constant__ int gpuCubeEdgeFlags[256];
bb_gpu __constant__ int gpuTriangleConnectionTable3D[256][16];

void initializeGPUTables(){
    CUCHECK(cudaMemcpyToSymbol(gpuEdgeConnection, edgeConnection, sizeof(int) * 12 * 2));
    CUCHECK(cudaMemcpyToSymbol(gpuCubeEdgeFlags, cubeEdgeFlags, sizeof(int) * 256));
    CUCHECK(cudaMemcpyToSymbol(gpuTriangleConnectionTable3D,
                               triangleConnectionTable3D, sizeof(int) * 256 * 16));
}

bb_cpu_gpu inline int EdgeFlags(int a){
#if defined(__CUDA_ARCH__)
    return gpuCubeEdgeFlags[a];
#else
    return cubeEdgeFlags[a];
#endif
}

bb_cpu_gpu inline int EdgeConnection(int a, int b){
#if defined(__CUDA_ARCH__)
    return gpuEdgeConnection[a][b];
#else
    return edgeConnection[a][b];
#endif
}

bb_cpu_gpu inline int ConnectionTable3D(int a, int b){
#if defined(__CUDA_ARCH__)
    return gpuTriangleConnectionTable3D[a][b];
#else
    return triangleConnectionTable3D[a][b];
#endif
}

template <typename T> bb_cpu_gpu
T distanceToZeroLevelSet(T phi0, T phi1) {
    if(std::fabs(phi0) + std::fabs(phi1) > Epsilon)
        return std::fabs(phi0) / (std::fabs(phi0) + std::fabs(phi1));
    return static_cast<T>(0.5);
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

bb_cpu_gpu
vec3f grad(FieldGrid3f *grid, size_t i, size_t j, size_t k, const vec3f &invSize){
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
    if(i > dimx - 2){
        ip = i;
    }else if(i == 0){
        im = 0;
    }

    if(j > dimy - 2){
        jp = j;
    }else if(j == 0){
        jm = 0;
    }

    if(k > dimz - 2){
        kp = k;
    }else if(k == 0){
        km = 0;
    }

    ret.x = 0.5f * invSize.x * (grid->GetValueAt(ip, j, k) - grid->GetValueAt(im, j, k));
    ret.y = 0.5f * invSize.y * (grid->GetValueAt(i, jp, k) - grid->GetValueAt(i, jm, k));
    ret.z = 0.5f * invSize.z * (grid->GetValueAt(i, j, kp) - grid->GetValueAt(i, j, km));
    return ret;
}

bb_cpu_gpu inline
vec3f safeNormalize(const vec3f& n){
    Float len = n.LengthSquared();
    if(len > 0.0 && !IsZero(len))
        return Normalize(n);
    return n;
}

// To compute unique edge ID, map vertices+edges into
// doubled virtual vertex indices.
//
// v  edge   v
// |----*----|    -->    |-----|-----|
// i        i+1         2i   2i+1  2i+2
//
bb_cpu_gpu inline
size_t globalEdgeId(size_t i, size_t j, size_t k, const vec3ui& dim,
                    size_t localEdgeId)
{
    // See edgeConnection in marching_cubes_table.h for the edge ordering.
    const int edgeOffset3D[12][3] = {
        {1, 0, 0}, {2, 0, 1}, {1, 0, 2}, {0, 0, 1}, {1, 2, 0}, {2, 2, 1},
        {1, 2, 2}, {0, 2, 1}, {0, 1, 0}, {2, 1, 0}, {2, 1, 2}, {0, 1, 2}
    };

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
bb_cpu_gpu inline
size_t globalVertexId(size_t i, size_t j, size_t k, const vec3ui& dim,
                      size_t localVertexId)
{
    // See edgeConnection in marching_cubes_table.h for the edge ordering.
    const int vertexOffset3D[8][3] = {{0, 0, 0}, {2, 0, 0}, {2, 0, 2},
                                      {0, 0, 2}, {0, 2, 0}, {2, 2, 0},
                                      {2, 2, 2}, {0, 2, 2}
    };

    return ((2 * k + vertexOffset3D[localVertexId][2]) * 2 * dim.y +
            (2 * j + vertexOffset3D[localVertexId][1])) * 2 * dim.x +
            (2 * i + vertexOffset3D[localVertexId][0]);
}

// Add cubeId to the marching cube work queue in case it needs further processing
bb_cpu_gpu
void buildMarchingCubeWorkQueue(Float *data, vec3f *normal, size_t *edges, Float isoValue,
                                size_t cubeId, Bounds3f bounds, MarchingCubeWorkQueue *mcQ)
{
    int indexFlag = 0;
    // Which vertices are inside? If i-th vertex is inside, mark '1' at i-th
    // bit. of 'idxFlagSize'
    for(int vertexId = 0; vertexId < 8; vertexId++){
        if(data[vertexId] <= isoValue){
            indexFlag |= 1 << vertexId;
        }
    }

    // If the cube is entirely inside or outside of the surface, there is no job
    // to be done in this marching-cube cell. In case there is, add it to
    // marching cube work queue.
    if(!(indexFlag == 0 || indexFlag == 255))
        mcQ->Push(MarchingCubeWorkItem(cubeId, indexFlag, bounds, data, normal, edges));
}

void MarchingCubes(FieldGrid3f *grid, HostTriangleMesh3 *mesh, Float isoValue,
                   bool rotate_faces)
{
    MarchingCubeVertexMap vertexMap;
    const vec3ui dim = grid->GetResolution();
    vec3f gridSize = grid->GetSpacing();

    const vec3f invGridSize(1.0 / (Float)gridSize.x,
                            1.0 / (Float)gridSize.y,
                            1.0 / (Float)gridSize.z);

    ssize_t dimx = static_cast<ssize_t>(dim.x);
    ssize_t dimy = static_cast<ssize_t>(dim.y);
    ssize_t dimz = static_cast<ssize_t>(dim.z);

    size_t zLength = dimz-1;
    size_t yLength = dimy-1;
    size_t xLength = dimx-1;
    ssize_t max_size = xLength * yLength * zLength;

    std::cout << " - Initializing GPU tables..." << std::flush;
    // Copy the marching cubes tables to gpu, otherwise we cannot run
    initializeGPUTables();

    std::cout << "done\n - Building marching cubes work queue..." << std::flush;
    int size = 0;
    int *requiredCubes = cudaAllocateUnregisterVx(int, 1);
    *requiredCubes = 0;

    // Make a first pass to compute the exact required cube count to perform this
    // reconstruction. Because marching cubes is a high resolution method it might
    // happen that we run out of memory on the work queue if we simply allocate
    // the maximum possible size, this way we fit the work queue to the required
    // execution.
    AutoParallelFor("MarchingCubes-DomainSize", (int)max_size, AutoLambda(int cubeId){
        Float data[8];
        size_t k = cubeId % zLength;
        size_t j = (cubeId / zLength) % yLength;
        size_t i = cubeId / (yLength * zLength);
        data[0] = grid->GetValueAt(i, j, k);
        data[1] = grid->GetValueAt(i + 1, j, k);
        data[4] = grid->GetValueAt(i, j + 1, k);
        data[5] = grid->GetValueAt(i + 1, j + 1, k);
        data[3] = grid->GetValueAt(i, j, k + 1);
        data[2] = grid->GetValueAt(i + 1, j, k + 1);
        data[7] = grid->GetValueAt(i, j + 1, k + 1);
        data[6] = grid->GetValueAt(i + 1, j + 1, k + 1);

        int indexFlag = 0;
        for(int vertexId = 0; vertexId < 8; vertexId++){
            if(data[vertexId] <= isoValue){
                indexFlag |= 1 << vertexId;
            }
        }

        if(!(indexFlag == 0 || indexFlag == 255))
            atomic_increase(requiredCubes);
    });

    size = *requiredCubes;

    if(size == 0){
        printf("done\n - Given SDF does not intersect surface, nothing to do ( 0 cells ).\n");
        cudaFree(requiredCubes);
        return;
    }

    // Prepare processing work queue for the cubes that actually need to be solved
    MarchingCubeWorkQueue *workQ = cudaAllocateUnregisterVx(MarchingCubeWorkQueue, 1);
    workQ->SetSlots((int)size, false);

    AutoParallelFor("MarchingCubes-BuildWorkQ", (int)max_size, AutoLambda(int cubeId){
        Float data[8];
        vec3f normals[8];
        size_t edgeIds[12];

        size_t k = cubeId % zLength;
        size_t j = (cubeId / zLength) % yLength;
        size_t i = cubeId / (yLength * zLength);

        // Compute points/normals/edges for this cube
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

        vec3ui _dim = grid->GetResolution();
        for(int e = 0; e < 12; e++)
            edgeIds[e] = globalEdgeId(i, j, k, _dim, e);

        Bounds3f bounds(grid->GetDataPosition(vec3ui(i, j, k)),
                        grid->GetDataPosition(vec3ui(i + 1, j + 1, k + 1)));

        buildMarchingCubeWorkQueue(data, normals, edgeIds, isoValue,
                                   cubeId, bounds, workQ);
    });

    // Make sure all slots are occupied
    AssureA(workQ->size == size, "Invalid build, bug?");

    // For each cube that needs to be solved we need to store some information
    // about vertices and indices for the final triangulation.
    MarchingCubeSingleCubeOutput *scOut =
            cudaAllocateUnregisterVx(MarchingCubeSingleCubeOutput, workQ->size * 5);

    // Initialize as all invalid
    for(int i = 0; i < workQ->size * 5; i++){
        scOut[i].cubeIndex = 0;
        scOut[i].valid = false;
    }

    std::cout << "done\n - Processing marching cubes cells..." << std::flush;
    // Process each cube. Compute the interpolated values on each edge and generate
    // the triangles required for this cell.
    AutoParallelFor("MarchingCubes-ProcessCubes", workQ->size, AutoLambda(int cubeIndex){
        int idxVertexOfTheEdge[2];
        int indexMap[8] = {0, 1, 5, 4, 2, 3, 7, 6};
        MarchingCubeWorkItem *mcItem = workQ->Ref(cubeIndex);
        vec3f *e = &mcItem->e[0];
        vec3f *n = &mcItem->n[0];
        int idxEdgeFlags = EdgeFlags(mcItem->idxFlagSize);

        // Find and store the point of intersection of the surface with each edge
        for(int itrEdge = 0; itrEdge < 12; itrEdge++){
            // If there is an intersection on this edge
            if(idxEdgeFlags & (1 << itrEdge)){
                idxVertexOfTheEdge[0] = EdgeConnection(itrEdge, 0);
                idxVertexOfTheEdge[1] = EdgeConnection(itrEdge, 1);

                vec3f pos0 = mcItem->bounds.Corner(indexMap[idxVertexOfTheEdge[0]]);
                vec3f pos1 = mcItem->bounds.Corner(indexMap[idxVertexOfTheEdge[1]]);

                vec3f normal0 = mcItem->normal[idxVertexOfTheEdge[0]];
                vec3f normal1 = mcItem->normal[idxVertexOfTheEdge[1]];

                Float phi0 = mcItem->data[idxVertexOfTheEdge[0]] - isoValue;
                Float phi1 = mcItem->data[idxVertexOfTheEdge[1]] - isoValue;

                Float alpha = distanceToZeroLevelSet(phi0, phi1);
                alpha = Clamp(alpha, 0.000001, 0.999999);

                e[itrEdge] = Lerp(pos0, pos1, alpha);
                n[itrEdge] = Lerp(normal0, normal1, alpha);
            }
        }

        // Store triangle information
        for(int itrTri = 0; itrTri < 5; itrTri++){
            if(ConnectionTable3D(mcItem->idxFlagSize, 3 * itrTri) < 0)
                break;

            MarchingCubeSingleCubeOutput *out = &scOut[5 * cubeIndex + itrTri];
            int k0 = 3 * itrTri + 0;
            int k1 = 3 * itrTri + 1;
            int k2 = 3 * itrTri + 2;

            out->valid = true;
            out->cubeIndex = cubeIndex;
            out->k[0] = k0;
            out->k[1] = k1;
            out->k[2] = k2;
            out->vKey[0] =
                    mcItem->edgeIds[ConnectionTable3D(mcItem->idxFlagSize, k0)];
            out->vKey[1] =
                    mcItem->edgeIds[ConnectionTable3D(mcItem->idxFlagSize, k1)];
            out->vKey[2] =
                    mcItem->edgeIds[ConnectionTable3D(mcItem->idxFlagSize, k2)];
        }
    });

    std::cout << "done\n - Assembling triangles..." << std::flush;
    // Final assembling needs to be done on CPU so we can compute
    // vertices indices without race conditions.
    for(int i = 0; i < workQ->size * 5; i++){
        MarchingCubeSingleCubeOutput *out = &scOut[i];
        if(!out->valid)
            continue;

        vec3ui face;
        MarchingCubeWorkItem *mcItem = workQ->Ref(out->cubeIndex);
        int idxFlagSize = mcItem->idxFlagSize;
        vec3f *e = &mcItem->e[0];
        vec3f *n = &mcItem->n[0];
        for(int j = 0; j < 3; j++){
            MarchingCubeVertexId vId;
            MarchingCubeVertexHashKey vKey = out->vKey[j];
            if(queryVertexId(vertexMap, vKey, &vId))
                face[j] = vId;
            else{
                int k = out->k[j];
                face[j] = mesh->numberOfPoints();
                mesh->addNormal(safeNormalize(n[ConnectionTable3D(idxFlagSize, k)]));
                mesh->addPoint(e[ConnectionTable3D(idxFlagSize, k)]);
                mesh->addUv(vec2f());
                vertexMap.insert(std::make_pair(vKey, face[j]));
            }
        }

        vec3ui actualFace = face;
        if(rotate_faces){
            actualFace[0] = face[0];
            actualFace[1] = face[2];
            actualFace[2] = face[1];
        }
        mesh->addPointUvNormalTriangle(actualFace, actualFace, actualFace);
    }

    cudaFree(workQ->ids);
    cudaFree(workQ);
    cudaFree(scOut);
    cudaFree(requiredCubes);
    std::cout << "done" << std::endl;
}

