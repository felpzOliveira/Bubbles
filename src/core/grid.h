#pragma once
#include <geometry.h>
#include <particle.h>
#include <sampling.h>

/*
* This is a minimal grid acceleration for SPH-based simulations,
* for Eulerian simulations need to implement a more robust grid.
*/

/*
* A Cell is a minimal representation of a Voxel which simulation domain
* is splitted into. It is made of 2 particle chains lists for fast particle
* splitting. At any time one of the lists is valid for hashing and the other
* one is used for updating after particles are updated, this way each Cell
* can update parallel without consulting all particles. Cell distribution
* update happens by checking neighboring cells and assuming a particle cannot
* in a single iteration move dx > h. This makes for a very fast update.
*/
template<typename T>
struct Cell{
    T bounds;
    ParticleChain *head, *headAux;
    ParticleChain *tail, *tailAux;
    int *neighborList;
    int neighborListCount;
    int id;
    int chainLength, chainAuxLength;
    int active;
    int level;
    
    __bidevice__ void Reset(){
        head = nullptr;
        tail = nullptr;
        headAux = nullptr;
        tailAux = nullptr;
        chainLength = 0;
        chainAuxLength = 0;
        active = 0;
        level = -1;
    }
    
    __bidevice__ void SetNeighborListPtr(int *list){
        neighborList = list;
    }
    
    __bidevice__ void SetNeighborList(int *list, int n){
        memcpy(neighborList, list, n * sizeof(int));
        neighborListCount = n;
    }
    
    __bidevice__ void SwapActive(){ active = 1 - active; }
    
    __bidevice__ int IsActive(){ return active; }
    
    __bidevice__ ParticleChain *GetChain(){
        if(active) return headAux;
        return head;
    }
    
    __bidevice__ ParticleChain *GetTail(){
        if(active) return tailAux;
        return tail;
    }
    
    __bidevice__ int GetChainLength(){
        if(active) return chainAuxLength;
        return chainLength;
    }
    
    
    __bidevice__ ParticleChain **GetActiveChain(){
        if(active) return &headAux;
        return &head;
    }
    
    __bidevice__ ParticleChain **GetActiveTail(){
        if(active) return &tailAux;
        return &tail;
    }
    
    
    __bidevice__ ParticleChain **GetNextChain(){
        if(!active) return &headAux;
        return &head;
    }
    
    __bidevice__ ParticleChain **GetNextTail(){
        if(!active) return &tailAux;
        return &tail;
    }
    
    __bidevice__ void IncreaseNextChainCount(){
        if(!active) chainAuxLength++;
        else chainLength ++;
    }
    
    __bidevice__ void IncreaseActiveChainCount(){
        if(active) chainAuxLength++;
        else chainLength ++;
    }
    
    __bidevice__ void ResetNext(){
        if(active){
            head = nullptr;
            tail = nullptr;
            chainLength = 0;
        }else{
            headAux = nullptr;
            tailAux = nullptr;
            chainAuxLength = 0;
        }
    }
    
    __bidevice__ void AddToNextChain(ParticleChain *node){
        ParticleChain **cHead = GetNextChain();
        ParticleChain **cTail = GetNextTail();
        if(!*cHead){
            *cHead = node;
            *cTail = node;
        }else{
            (*cTail)->next = node;
            (*cTail) = node;
        }
        
        (*cTail)->next = nullptr;
        IncreaseNextChainCount();
    }
    
    __bidevice__ void AddToChain(ParticleChain *node){
        ParticleChain **cHead = GetActiveChain();
        ParticleChain **cTail = GetActiveTail();
        if(!*cHead){
            *cHead = node;
            *cTail = node;
        }else{
            (*cTail)->next = node;
            (*cTail) = node;
        }
        
        (*cTail)->next = nullptr;
        IncreaseActiveChainCount();
    }
    
    __bidevice__ void SetLevel(int L){
        level = L;
    }
    
    __bidevice__ int GetLevel(){
        return level;
    }
    
    __bidevice__ void Set(T b, int _id){
        bounds = b;
        id = _id;
        Reset();
    }
};

typedef Cell<Bounds1f> Cell1;
typedef Cell<Bounds2f> Cell2;
typedef Cell<Bounds3f> Cell3;

template<typename U> inline __bidevice__
U DimensionalIndex(unsigned int id, const U &usizes, int dimensions){
    AssertA(dimensions == 1 || dimensions == 2 || dimensions == 3, 
            "Unknown dimension distribution");
    U u;
    u[0] = id;
    if(dimensions == 3){
        int plane = usizes[0] * usizes[1];
        u[2] = (int)(id / plane);
        id -= u[2] * plane;
    }
    
    if(dimensions > 1){
        u[1] = (int)(id / usizes[0]);
        u[0] = (int)(id % usizes[0]);
    }
    
    return u;
}

template<typename U> inline __bidevice__ 
unsigned int LinearIndex(const U &u, const U &usizes, int dimensions){
    AssertA(dimensions == 1 || dimensions == 2 || dimensions == 3, 
            "Unknown dimension distribution {GetLinearIndex}");
    
    unsigned int h = u[0]; // x
    if(dimensions > 1)
        h += u[1] * usizes[0]; // y * sizeX
    if(dimensions == 3)
        h += (u[2] * usizes[0] * usizes[1]); // z * sizeX * sizeY
    return h;
}

// Vector computation  Dimension computation  Domain computation 
// T = vec2f/vec3f,    U = vec2ui/vec3ui,     Q = Bounds2f/Bounds3f
template<typename T, typename U, typename Q>
class Grid{
    public:
    U usizes; // the amount of cells in each dimension
    unsigned int total; // the amount of cells in total
    Cell<Q> *cells; // cells in this grid
    T cellsLen; // cells length in each dimension
    T minPoint; // grid minimal point for hashing
    int dimensions; // checker for dimension
    Q bounds; // the bounds of this grid after construction
    int maxLevels; // in case LNM was executed, mark the max level value {unsafe}
    int indicator; // flag for GPU-domain based operations
    int *neighborListPtr; // address of the first neighborList
    int *activeCells; // list of cells that actually have particle in them
    int activeCellsCount; // amount of active cells at any given moment
    
    __bidevice__ Grid(){ SetDimension(T(0)); }
    __bidevice__ void SetDimension(const Float &u){ (void)u; dimensions = 1; }
    __bidevice__ void SetDimension(const vec2f &u){ (void)u; dimensions = 2; }
    __bidevice__ void SetDimension(const vec3f &u){ (void)u; dimensions = 3; }
    __bidevice__ void SetLNMMaxLevel(const int &level){ maxLevels = level; }
    __bidevice__ int GetLNMMaxLevel() { return maxLevels; }
    __bidevice__ Q GetBounds(){ return bounds; }
    __bidevice__ T GetCellSize(){ return cellsLen; }
    __bidevice__ int GetDimensions(){ return dimensions; }
    __bidevice__ unsigned int GetCellCount(){ return total; }
    __bidevice__ int GetActiveCellCount(){ return activeCellsCount; }
    __bidevice__ U GetIndexCount(){ return usizes; }
    __bidevice__ Float GetCellSizeOn(int axis){
        AssertA(axis >= 0 && axis < dimensions, "Invalid axis given for CellSizeOn");
        return cellsLen[axis];
    }
    __bidevice__ unsigned int GetCountOn(int axis){
        AssertA(axis >= 0 && axis < dimensions, "Invalid axis given for CountOn");
        return usizes[axis];
    }
    
    __bidevice__ int GetCellLevel(int cellId){
        AssertA(cellId >= 0 && cellId < total, "Invalid cellId for GetCellLevel");
        Cell<Q> *cell = &cells[cellId];
        return cell->GetLevel();
    }
    
    __bidevice__ int GetActiveCellId(int qid){
        if(!(qid >= 0 && qid < activeCellsCount)){
            printf("Got query for %d but have only %d\n", qid, activeCellsCount);
        }
        AssertA(qid >= 0 && qid < activeCellsCount, "Invalid cellId for GetActiveCellId");
        return activeCells[qid];
    }
    
    __bidevice__ Cell<Q> *GetActiveCell(int qid){
        AssertA(qid >= 0 && qid < activeCellsCount, "Invalid cellId for GetActiveCell");
        return &cells[activeCells[qid]];
    }
    
    __bidevice__ Cell<Q> *GetCell(int cellId){
        AssertA(cellId >= 0 && cellId < total, "Invalid cellId for GetCell");
        return &cells[cellId];
    }
    
    /* Get offset for dimension value 'p' on axis 'axis' in case it lies on boundary */
    __bidevice__ Float ExtremeEpsilon(Float p, int axis){
        Float eps = 0;
        Float p0 = bounds.LengthAt(0, axis);
        Float p1 = bounds.LengthAt(1, axis);
        if(IsZero(p - p0)){
            eps = Epsilon;
        }else if(IsZero(p - p1)){
            eps = -Epsilon;
        }
        
        return eps;
    }
    
    /* Hash position 'p' into a cell index */
    __bidevice__ U GetHashedPosition(const T &p){
        U u;
        if(!Inside(p, bounds)){
            printf(" [ERROR] : Requested for hash on point outside domain ");
            p.PrintSelf();
            printf(" , Bounds: ");
            bounds.PrintSelf();
            printf("\n");
        }
        
        AssertA(Inside(p, bounds), "Out of bounds point");
        for(int i = 0; i < dimensions; i++){
            Float pi = p[i];
            pi += ExtremeEpsilon(pi, i);
            
            Float dmin = minPoint[i];
            Float dlen = cellsLen[i];
            Float dp = (pi - dmin) / dlen;
            
            int linearId = (int)(Floor(dp));
            AssertA(linearId >= 0 && linearId < usizes[i], "Out of bounds position");
            u[i] = linearId;
        }
        
        return u;
    }
    
    /* Get logical index of cell */
    __bidevice__ U GetCellIndex(unsigned int i){
        return DimensionalIndex(i, usizes, dimensions);
    }
    
    /* Get the ordered cell index of a cell */
    __bidevice__ unsigned int GetLinearCellIndex(const U &u){
        unsigned int h = LinearIndex(u, usizes, dimensions);
        AssertA(h < total, "Invalid cell id computation");
        return h;
    }
    
    /* Hash position 'p' and get the linear cell index */
    __bidevice__ unsigned int GetLinearHashedPosition(const T &p){
        U u = GetHashedPosition(p);
        return GetLinearCellIndex(u);
    }
    
    /* Distribute routines for multiple particle types, low level control */
    __bidevice__ void DistributeResetCell(unsigned int cellId){
        AssertA(cellId < total, "Invalid distribution cell id");
        Cell<Q> *cell = &cells[cellId];
        cell->Reset();
    }
    
    /* Distribute to specific cell, low level */
    template<typename ParticleType = ParticleSet<T>>
        __bidevice__ void DistributeAddToCell(ParticleType *pSet, unsigned int cellId){
        AssertA(cellId < total, "Invalid distribution cell id");
        Cell<Q> *cell = &cells[cellId];
        /* Insert particles [that belong here] to this chain */
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            T p = pSet->GetParticlePosition(i);
            U u = GetHashedPosition(p);
            unsigned int h = GetLinearCellIndex(u);
            if(h == cellId){
                AssertA(Inside(p, cell->bounds), "Invalid cell computation {Inside}");
                ParticleChain *pChain = pSet->GetParticleChainNode(i);
                AssertA(pChain, "Invalid particle chain node");
                
                pChain->cId = cellId;
                pChain->pId = i;
                pChain->sId = pSet->GetFamilyId();
                cell->AddToChain(pChain);
            }
        }
    }
    
    /* 
    * Distribute a list of particles to the cell in CPU. In order to avoid
    * performing full distribution if a continuous emission is being used
    * allow for distributing a list of particles.
    */
    template<typename ParticleType = ParticleSet<T>>
        __host__ void DistributeByParticleList(ParticleType *pSet, T *pList, 
                                               int pCount, int startId,
                                               Float kernelRadius)
    {
        for(int i = 0; i < pCount; i++){
            ParticleChain *pChain;
            T p = pList[i];
            U u = GetHashedPosition(p);
            unsigned int h = GetLinearCellIndex(u);
            AssertA(h < total, "Invalid particle position");
            Cell<Q> *cell = &cells[h];
            AssertA(Inside(p, cell->bounds), "Invalid particle computation {Inside}");
            
            if(cell->IsActive()){
                pChain = pSet->GetParticleAuxChainNode(startId+i);
            }else{
                pChain = pSet->GetParticleChainNode(startId+i);
            }
            
            pChain->cId = h;
            pChain->pId = startId+i;
            pChain->sId = pSet->GetFamilyId();
            cell->AddToChain(pChain);
        }
        
        for(int i = 0; i < pCount; i++){
            DistributeParticleBucket(pSet, startId+i, kernelRadius);
        }
    }
    
    /* Distribute by particle, faster for initialization. Can only run on CPU */
    template<typename ParticleType = ParticleSet<T>>
        __host__ void DistributeByParticle(ParticleType *pSet){
        int pCount = pSet->GetParticleCount();
        for(int i = 0; i < pCount; i++){
            T p = pSet->GetParticlePosition(i);
            U u = GetHashedPosition(p);
            unsigned int h = GetLinearCellIndex(u);
            AssertA(h < total, "Invalid particle position");
            Cell<Q> *cell = &cells[h];
            AssertA(Inside(p, cell->bounds), "Invalid particle computation {Inside}");

            ParticleChain *pChain = pSet->GetParticleChainNode(i);
            pChain->cId = h;
            pChain->pId = i;
            pChain->sId = pSet->GetFamilyId();
            cell->AddToChain(pChain);
        }
    }
    
    /* Distribute particles from given ParticleSet by cell Id with optmized search */
    template<typename ParticleType = ParticleSet<T>>
        __bidevice__ void DistributeToCellOpt(ParticleType *pSet, unsigned int cellId){
        DistributeToCellOpt(&pSet, 1, cellId);
    }
    
    /* Distribute particles from given ParticleSet by cell Id */
    template<typename ParticleType = ParticleSet<T>>
        __bidevice__ void DistributeToCell(ParticleType *pSet, unsigned int cellId){
        DistributeResetCell(cellId);
        DistributeAddToCell(pSet, cellId);
    }
    
    template<typename ParticleType = ParticleSet<T>>
        __bidevice__ void DistributeParticleBucket(ParticleType *pSet, int pid,
                                                   Float kernelRadius)
    {
        T pi = pSet->GetParticlePosition(pid);
        unsigned int cellId = GetLinearHashedPosition(pi);
        int *neighbors = nullptr;
        int count = GetNeighborsOf(cellId, &neighbors);
        Bucket *bucket = pSet->GetParticleBucket(pid);
        bucket->Reset();
        for(int i = 0; i < count; i++){
            Cell<Q> *cell = GetCell(neighbors[i]);
            ParticleChain *pChain = cell->GetChain();
            int size = cell->GetChainLength();
            
            for(int j = 0; j < size; j++){
                T pj = pSet->GetParticlePosition(pChain->pId);
                Float distance = Distance(pi, pj);
                if(IsWithinStd(distance, kernelRadius)){
                    bucket->Insert(pChain->pId);
                }
                
                pChain = pChain->next;
            }
        }
    }
    
    template<typename ParticleType = ParticleSet<T>>
        __bidevice__ void DistributeToCellOpt(ParticleType **ppSet, int n, 
                                              unsigned int cellId)
    {
        AssertA(cellId < total, "Invalid distribution cell id");
        int *neighbors = nullptr;
        Cell<Q> *cell = &cells[cellId];
        cell->ResetNext();
        int count = GetNeighborsOf(cellId, &neighbors);
        for(int i = 0; i < count; i++){
            Cell<Q> *neighbor = &cells[neighbors[i]];
            int size = neighbor->GetChainLength();
            ParticleChain *pChain = neighbor->GetChain();
            for(int j = 0; j < size; j++){
                AssertA(pChain != NULL, "Unstabble simulation, not a valid chain configuration");
                ParticleType *pSet = NULL;
                for(int i = 0; i < n; i++){
                    pSet = ppSet[i];
                    if(pSet->GetFamilyId() == pChain->sId) break;
                }
                
                T p = pSet->GetParticlePosition(pChain->pId);
                U u = GetHashedPosition(p);
                
                unsigned int h = GetLinearCellIndex(u);
                if(h == cellId){
                    AssertA(Inside(p, cell->bounds), "Invalid cell computation {Inside}");
                    ParticleChain *pToAdd;
                    if(cell->IsActive()){
                        pToAdd = pSet->GetParticleChainNode(pChain->pId);
                    }else{
                        pToAdd = pSet->GetParticleAuxChainNode(pChain->pId);
                    }
                    
                    AssertA(pToAdd, "Invalid particle chain node");
                    
                    pToAdd->cId = cellId;
                    pToAdd->pId = pChain->pId;
                    pToAdd->sId = pSet->GetFamilyId();
                    cell->AddToNextChain(pToAdd);
                }
                
                pChain = pChain->next;
            }
        }
    }
    
    __bidevice__ void SwapCellList(int cellId){
        AssertA(cellId < total, "Invalid distribution cell id");
        Cell<Q> *cell = &cells[cellId];
        cell->SwapActive();
    }
    
    /* Query the cell neighbor list */
    __bidevice__ int GetNeighborsOf(int id, int **neighbors){
        AssertA(id < total, "Invalid cell id given for {GetNeighborsOf}");
        Cell<Q> *cell = &cells[id];
        *neighbors = cell->neighborList;
        return cell->neighborListCount;
    }
    
    
    /* Get neighbors from a cell with a depth 'depth' */
    __bidevice__ int GetNeighborListFor(int id, int depth, int *neighbors){
        AssertA(dimensions == 1 || dimensions == 2 || dimensions == 3, 
                "Unknown dimension distribution {GetNeighborListFor}");
        int count = 0;
        U u = GetCellIndex(id);
        T ufmin, ufmax;
        for(int s = 0; s < dimensions; s++){
            ufmin[s] = u[s];
            ufmax[s] = u[s];
        }
        ufmin = ufmin - T(depth);
        ufmax = ufmax + T(depth);
        
        for(int j = ufmin[1]; j <= ufmax[1]; j++){
            if(j < 0 || j >= usizes[1]) continue;
            if(dimensions > 1){
                for(int i = ufmin[0]; i <= ufmax[0]; i++){
                    if(i < 0 || i >= usizes[0]) continue;
                    if(dimensions == 3){
                        for(int k = ufmin[2]; k <= ufmax[2]; k++){
                            if(k < 0 || k >= usizes[2]) continue;
                            U f; f[0] = i; f[1] = j; f[2] = k;
                            int fid = GetLinearCellIndex(f);
                            neighbors[count++] = fid;
                        }
                    }else{
                        U f; f[0] = i; f[1] = j;
                        int fid = GetLinearCellIndex(f);
                        neighbors[count++] = fid;
                    }
                }
            }else{
                U f; f[0] = j;
                int fid = GetLinearCellIndex(f);
                neighbors[count++] = fid;
            }
        }
        return count;
    }
    
    /* Computes and allocates the cells for this grid */
    __host__ void Build(const U &resolution, const T &dp0, const T &dp1);
    
    /* Sets and compute grid usage */
    __host__ void UpdateQueryState();
};

template<typename T, typename U, typename Q>
__global__ void BuildNeighborListKernel(Grid<T, U, Q> *grid){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->total){
        int neighbor[27];
        int count;
        Cell<Q> *cell = &grid->cells[i];
        U u = grid->GetCellIndex(i);
        T center;
        for(int k = 0; k < grid->dimensions; k++){
            center[k] = grid->minPoint[k] + u[k] * grid->cellsLen[k] + 
                0.5 * grid->cellsLen[k];
        }
        
        T pMin = center - 0.5 * grid->cellsLen;
        T pMax = center + 0.5 * grid->cellsLen;
        
        count = grid->GetNeighborListFor(i, 1, &neighbor[0]);
        cell->SetNeighborListPtr(&grid->neighborListPtr[27 * i]);
        cell->Set(Q(pMin, pMax), i);
        cell->SetNeighborList(&neighbor[0], count);
    }
}

template<typename T, typename U, typename Q>
__host__ void Grid<T, U, Q>::Build(const U &resolution, const T &dp0, const T &dp1){
    SetDimension(T(0));
    T lower(Infinity), high(-Infinity);
    T p0 = Max(dp0, dp1);
    T p1 = Min(dp0, dp1);
    T maxPoint;
    total = 1;
    for(int k = 0; k < dimensions; k++){
        Float dl = p0[k];
        Float du = p1[k];
        if(dl < lower[k]) lower[k] = dl;
        if(dl > high[k]) high[k] = dl;
        if(du > high[k]) high[k] = du;
        if(du < lower[k]) lower[k] = du;
    }
    
    for(int k = 0; k < dimensions; k++){
        Float s = high[k] - lower[k];
        Float len = s / (Float)resolution[k];
        cellsLen[k] = len;
        usizes[k] = (int)std::ceil(s / len);
        maxPoint[k] = lower[k] + (Float)usizes[k] * cellsLen[k];
        total *= usizes[k];
    }
    
    minPoint = lower;
    bounds = Q(minPoint, maxPoint);
    
    cells = cudaAllocateVx(Cell<Q>, total);
    neighborListPtr = cudaAllocateVx(int, total * 27);
    activeCells = cudaAllocateVx(int, total);
    activeCellsCount = 0;
    
    printf("Building acceleration query list [%d] ... ", total);
    fflush(stdout);
    
    GPULaunch(total, GPUKernel(BuildNeighborListKernel<T, U, Q>), this);
    
    printf("OK\n");
}

template<typename T, typename U, typename Q>
__host__ void Grid<T, U, Q>::UpdateQueryState(){
    int it = 0;
    for(int i = 0; i < total; i++){
        Cell<Q> *cell = &cells[i];
        if(cell->GetChainLength() > 0){
            activeCells[it++] = i;
        }
        
        cell->SetLevel(-1);
    }
    
    activeCellsCount = it;
}

typedef Grid<vec1f, vec1ui, Bounds1f> Grid1;
typedef Grid<vec2f, vec2ui, Bounds2f> Grid2;
typedef Grid<vec3f, vec3ui, Bounds3f> Grid3;


template<typename T, typename U, typename Q>
__host__ void ResetAndDistribute(Grid<T, U, Q> *grid, ParticleSet<T> *pSet){
    if(grid && pSet){
        for(int i = 0; i < grid->GetCellCount(); i++){
            grid->DistributeResetCell(i);
        }
        
        grid->DistributeByParticle(pSet);
    }
}

inline __host__ Grid2 *MakeGrid(const vec2ui &size, const vec2f &pMin, const vec2f &pMax){
    Grid2 *grid = cudaAllocateVx(Grid2, 1);
    grid->Build(size, pMin, pMax);
    return grid;
}

inline __host__ Grid3 *MakeGrid(const vec3ui &size, const vec3f &pMin, const vec3f &pMax){
    Grid3 *grid = cudaAllocateVx(Grid3, 1);
    grid->Build(size, pMin, pMax);
    return grid;
}

/*
* This is a grid abstraction for Grid-based simulation, we keep a Grid pointer here
* so that faster particle-cell relation can be made with the Grid hashing support.
* Add values so that interpolation particle-cell/cell-particle can be performed on
* different fields.
*/
// Vector computation  Dimension computation  Domain computation           Values
// T = vec2f/vec3f,    U = vec2ui/vec3ui,     Q = Bounds2f/Bounds3f,  F = Float/vec2f/vec3f
template<typename T, typename U, typename Q, typename F>
class NodeEdgeGrid{
    public:
    Grid<T, U, Q> *grid; // the underlying grid structure for hashing
    /*
    * Field values
    */
    F *data; // raw values
    U usizes; // node count in each direction
    int totalCount; // total count of nodes
    int nodesPerCell; // amount of nodes per cell
    int gridDim; // grid dimension
    
    __bidevice__ NodeEdgeGrid(){
        grid = nullptr; data = nullptr;
        totalCount = 0; nodesPerCell = 0;
    }
    
    __bidevice__ int GetNodeCount(){ return totalCount; }
    __bidevice__ int GetDimensions(){ return gridDim; }
    __bidevice__ T GetSpacing(){ return grid->GetCellSize(); }
    __bidevice__ U GetNodeIndexCount(){ return usizes; }
    __bidevice__ int GetNodesPerCell(){ return nodesPerCell; }
    
    /* Get Cell from index *INDEX IS FROM GRID* */
    __bidevice__ Cell<Q> *GetCell(unsigned int id){
        return grid->GetCell(id);
    }
    
    /* 
    * Initializes the NodeEdgeGrid with the underlying grid,
    * the grid needs to be already built.
    */
    __host__ void Build(Grid<T, U, Q> *gridPtr, F initialValue = F(0)){
        AssertA(gridPtr, "Invalid grid pointer given for {NodeEdgeGrid::Setup}");
        // Get grid properties
        U ucount = gridPtr->GetIndexCount();
        gridDim = gridPtr->GetDimensions();
        
        // compute amount of nodes required for this grid
        totalCount = 1;
        nodesPerCell = 1;
        for(int i = 0; i < gridDim; i++){
            usizes[i] = (ucount[i] + 1);
            totalCount *= usizes[i];
            nodesPerCell *= 2;
        }
        
        // get memory
        data = cudaAllocateVx(F, totalCount);
        
        // initialize
        for(int i = 0; i < totalCount; i++) data[i] = initialValue;
        grid = gridPtr;
    }
    
    /* Get linear index of a node */
    __bidevice__ unsigned int GetNodeLinearIndex(const U &u){
        return LinearIndex(u, usizes, gridDim);
    }
    
    /* Get index for a specific node */
    __bidevice__ U GetNodeIndex(unsigned int nodeId){
        return DimensionalIndex(nodeId, usizes, gridDim);
    }
    
    /* Get cells that contains a node */ // TODO: Review for 3D
    __bidevice__ int GetCellsFrom(unsigned int nodeId, int *cells){
        AssertA(gridDim == 2, "Unsupported routine");
        int count = 0;
        U u = DimensionalIndex(nodeId, usizes, gridDim);
        U size = grid->GetIndexCount();
        U u1 = u - U(1);
        if(u[0] < size[0] && u[1] < size[1]){
            cells[count++] = grid->GetLinearCellIndex(u);
        }
        
        if(u1[0] < size[0] && u1[1] < size[1]){
            cells[count++] = grid->GetLinearCellIndex(u1);
        }
        
        for(int i = 0; i < gridDim; i++){
            U tmp = u;
            tmp[i] -= 1;
            if(tmp[0] < size[0] && tmp[1] < size[1]){
                cells[count++] = grid->GetLinearCellIndex(tmp);
            }
        }
        
        return count;
    }
    
    /* Get the nodes from a given a cell */ // TODO: Review for 3D
    __bidevice__ int GetNodesFrom(unsigned int cellId, int *nodes){
        AssertA(gridDim == 2, "Unsupported routine");
        // compute d-dimension index, this is the same as the first node
        U u = grid->GetCellIndex(cellId);
        nodes[0] = (u[0] + 0) + (u[1] + 0) * usizes[0]; // lower left node
        nodes[1] = (u[0] + 1) + (u[1] + 0) * usizes[0]; // lower right node
        nodes[2] = (u[0] + 0) + (u[1] + 1) * usizes[0]; // upper left node
        nodes[3] = (u[0] + 1) + (u[1] + 1) * usizes[0]; // upper right node
        return nodesPerCell;
    }
    
    /* Get field value for the given linear index */
    __bidevice__ F GetValue(unsigned int u){
        AssertA(u < totalCount, "Invalid index for {NodeEdgeGrid::GetValue}");
        return data[u];
    }
    
    /* Get field value for the given index */
    __bidevice__ F GetValue(const U &u){
        unsigned int h = LinearIndex(u, usizes, gridDim);
        AssertA(h < totalCount, "Invalid index for {NodeEdgeGrid::GetValue}");
        return data[h];
    }
    
    /* Set field value at a particular index */
    __bidevice__ void SetValue(const U &u, F value){
        unsigned int h = LinearIndex(u, usizes, gridDim);
        AssertA(h < totalCount, "Invalid index for {NodeEdgeGrid::SetValue}");
        data[h] = value;
    }
    
    /* Set field value at a particular linear index */
    __bidevice__ void SetValue(unsigned int u, F value){
        AssertA(u < totalCount, "Invalid index for {NodeEdgeGrid::SetValue}");
        data[u] = value;
    }
    
    /* Distribute the value of a particle to the node given by 'nodeId' */ 
    __bidevice__ void ParticleToNodes(const T &pos, F value, unsigned int nodeId){
        AssertA(nodeId < totalCount, "Invalid index for {NodeEdgeGrid::ParticleToNode}");
        U u = DimensionalIndex(nodeId, usizes, gridDim); // get d-dimensional index
        ParticleToNodes(pos, value, u, nodeId);
    }
    
    /* Distribute the value of a particle to the node given by 'u' */ 
    __bidevice__ void ParticleToNodes(const T &pos, F value, const U &u){
        unsigned nodeId = LinearIndex(u, usizes, gridDim);
        ParticleToNodes(pos, value, u, nodeId);
    }
    
    /* Distribute the value of a particle to the node given by 'u' and 'nodeId' */ 
    __bidevice__ void ParticleToNodes(const T &pos, F value, const U &u, unsigned int nodeId){
        T nP;
        for(int i = 0; i < gridDim; i++) nP[i] = u[i];
        T h = GetSpacing(); // spacing used in this grid
        nP = nP * h + grid->minPoint;
        /* NOTE: The distance vector must be normalized for [0,1] in each axis
        * and the value for the node depends on the "diagonally-opposed" volume
        * this way the closest a particle is to a node the volume increases
        * and so the contribution, so we get each weight as: 1.0 - di/hi,
        * di/hi normalizes the axis and the 1.0 invert the weight to the opposed size.
        */
        Float vol = 1; // gether the opposed volume/area occupied by the positive vector
        T d = Abs(nP - pos);
        for(int i = 0; i < gridDim; i++) vol *= (1.0 - d[i] / h[i]);
        if(!(vol > 0 || IsZero(vol)) && 0){
            T s;
            s.x = d.x / h.x;
            s.y = d.y / h.y;
            printf("Invalid Volume: %g {%g %g} - {%g %g} d = {%g %g} | {%g %g}\n", vol, nP.x, 
                   nP.y, pos.x, pos.y, d.x, d.y, s.x, s.y);
        }
        //AssertA(vol > 0 || IsZero(vol), 
        //"Reference particle outside cell domain during {ParticleToNodes}");
        data[nodeId] += vol * value; // value is ditributed according to its volume
    }
    
    /* Distribute the node values to obtain interpolated particle value */
    __bidevice__ F NodesToParticle(const T &pos){
        int nodes[8]; // since we only deal with max 3D 8 is the maximum nodes
        unsigned int cellId = grid->GetLinearHashedPosition(pos);
        int count = GetNodesFrom(cellId, &nodes[0]);
        T h = GetSpacing(); // spacing used in this grid
        /* NOTE: Do not confuse things. This operation does not reverse 
        * the 'ParticleToNodes' computation. It performs a interpolation
        * between the 4 nodes to the particle based on the volume referenced
        * by each node, i.e.: NodesToParticle^-1 != ParticleToNodes
*/
        F value = 0;
        for(int i = 0; i < count; i++){
            T nP;
            U u = DimensionalIndex(nodes[i], usizes, gridDim);
            for(int k = 0; k < gridDim; k++) nP[k] = u[k];
            nP = nP * h + grid->minPoint;
            T d = Abs(nP - pos);
            Float vol = 1;
            for(int k = 0; k < gridDim; k++) vol *= (1.0 - d[k] / h[k]);
            if(!(vol > 0 || IsZero(vol)) && 0){
                T s;
                s.x = d.x / h.x;
                s.y = d.y / h.y;
                printf("Invalid Volume: %g {%g %g} - {%g %g} d = {%g %g} | {%g %g}\n", vol, nP.x, 
                       nP.y, pos.x, pos.y, d.x, d.y, s.x, s.y);
            }
            //AssertA(vol > 0 || IsZero(vol), 
            //"Reference particle outside cell domain during {NodesToParticle}");
            value += vol * GetValue(nodes[i]);
        }
        
        return value;
    }
    
    /* Compute Node Volume */
    __bidevice__ Float NodeVolume(unsigned int id){
        AssertA(id < totalCount, "Invalid index for {NodeEdgeGrid::NodeVolume}");
        U u = DimensionalIndex(id, usizes, gridDim);
        Float V = 1;
        T h = GetSpacing();
        for(int i = 0; i < gridDim; i++){
            V *= h[i];
            if(u[i] == 0 || u[i] == usizes[i]-1) V *= 0.5;
        }
        
        return V;
    }
};


typedef NodeEdgeGrid<vec2f, vec2ui, Bounds2f, Float> NodeEdgeGrid2f;
typedef NodeEdgeGrid<vec2f, vec2ui, Bounds2f, vec2f> NodeEdgeGrid2v;

/*
* Because NodeEdgeGrid is made for particle <=> grid data transfers
* it might be too complex for a few operations. The Grid structure also
* is made for particles, we need a simpler version of Grid that can hold
* components based on edge/center locations and allow interpolation. Enters FieldGrid.
*/

// Vector computation  Dimension computation  Domain computation          Field Values
// T = vec2f/vec3f,    U = vec2ui/vec3ui,     Q = Bounds2f/Bounds3f,  F = Float/vec2f/vec3f
template<typename T, typename U, typename Q, typename F>
class FieldGrid{
    public:
    // the amount of elements in each direction for VertexCentered and CellCentered
    // the amount of cells for FaceCentered
    U resolution;
    
    unsigned int total; // the total amount of elements
    // the actual value stored in each node for VertexCentered and CellCentered
    // 1D array of (u,v,w) data for FaceCentered
    F *field;
    F *fieldUVW[3]; // easier access
    int perComponent[3]; // easier access to per component count
    
    T minPoint; // minimal point (bottom-left)
    T spacing; // spacing between nodes
    Q bounds; // bounds of the grid
    int dimensions; // number of dimensions of the grid
    VertexType type; // where should position hash to
    int filled; // flag indicating if this grid has values set
    __bidevice__ FieldGrid(){ SetDimension(T(0)); }
    __bidevice__ void SetDimension(const Float &u){ (void)u; dimensions = 1; }
    __bidevice__ void SetDimension(const vec2f &u){ (void)u; dimensions = 2; }
    __bidevice__ void SetDimension(const vec3f &u){ (void)u; dimensions = 3; }
    __bidevice__ int Filled(){ return filled; }
    __bidevice__ void MarkFilled(){ filled = 1; }
    
    __bidevice__ U GetComponentDimension(int component){
        AssertA(component < dimensions, "Invalid component dimension");
        U size(0);
        size[component] = resolution[component] + 1;
        for(int i = 0; i < dimensions; i++){
            if(i != component){
                size[i] = resolution[i];
            }
        }
        
        return size;
    }
    
    __bidevice__ T GetDataPosition(const U &index, int component){
        T res(0);
        AssertA(type == FaceCentered, "Incorrect query");
        if(component < dimensions){
            T origin(0);
            T gridSpacing = spacing;
            gridSpacing[component] = 0;
            
            origin = minPoint + 0.5 * gridSpacing;
            for(int i = 0; i < dimensions; i++){
                res[i] = origin[i] + spacing[i] * index[i];
            }
        }
        
        return res;
    }
    
    __bidevice__ T GetDataPosition(const U &index){
        T res(0);
        T origin(0);
        AssertA(type != FaceCentered, "Incorrect query");
        switch(type){
            case VertexCentered: origin = minPoint; break;
            case CellCentered: origin = minPoint + 0.5 * spacing; break;
            default:{
                printf("Unimplemented FieldGrid type\n");
            }
        }
        
        for(int i = 0; i < dimensions; i++){
            res[i] = origin[i] + spacing[i] * index[i];
        }
        
        return res;
    }
    
    /*
    * Returns the amount of data points required to represent a type of FieldGrid,
    * i.e.: VertexCentered for example requires n+1 points while CellCentered only n.
*/
    //TODO: Implement other types
    __bidevice__ unsigned int Get1DLengthFor(int count){
        switch(type){
            case VertexCentered: return count+1;
            case CellCentered: return count;
            default:{
                printf("Unknown grid node distribution\n");
                return 0;
            }
        }
    }
    
    __bidevice__ void SetValueAt(const F &value, const U &u){
        unsigned int h = LinearIndex(u, resolution, dimensions);
        field[h] = value;
    }
    
    __bidevice__ F GetValueAt(const vec3ui &u){
        vec3ui r(resolution[0], resolution[1], resolution[2]);
        unsigned int h = LinearIndex<vec3ui>(u, r, dimensions);
        return field[h];
    }
    
    __bidevice__ F GetValueAt(const vec2ui &u){
        vec2ui r(resolution[0], resolution[1]);
        unsigned int h = LinearIndex<vec2ui>(u, r, dimensions);
        return field[h];
    }
    
    /*
    * Sample the field in the given point p.
*/
    __bidevice__ F Sample(const T &p){
        U ii(0);
        U jj(0);
        T weight(0);
        T origin = GetDataPosition(U(0));
        T normalized = p - origin;
        for(int i = 0; i < dimensions; i++){
            int id;
            Float f;
            AssertA(!IsZero(spacing[i]), "Zero spacing");
            normalized[i] /= spacing[i];
            AssertA(!normalized.HasNaN(), "NaN normalized position");
            GetBarycentric(normalized[i], 0, resolution[i]-1, &id, &f);
            weight[i] = f;
            AssertA(!IsNaN(f), "NaN at barycentric weight");
            ii[i] = id;
            jj[i] = Min(id+1, resolution[i]-1);
        }
        
        if(dimensions == 3){
            return Trilerp(
                           GetValueAt(ii), // (i,j,k)
                           GetValueAt(vec3ui(jj[0], ii[1], ii[2])), // (i+1,j,k)
                           GetValueAt(vec3ui(ii[0], jj[1], ii[2])), // (i,j+1,k)
                           GetValueAt(vec3ui(jj[0], jj[1], ii[2])), // (i+1,j+1,k)
                           GetValueAt(vec3ui(ii[0], ii[1], jj[2])), // (i,j,k+1)
                           GetValueAt(vec3ui(jj[0], ii[1], jj[2])), // (i+1,j,k+1)
                           GetValueAt(vec3ui(ii[0], jj[1], jj[2])), // (i,j+1,k+1)
                           GetValueAt(jj),  // (i+1,j+1,k+1)
                           weight[0], weight[1], weight[2]);
        }else{
            return Bilerp(
                          GetValueAt(ii), // (i,j)
                          GetValueAt(vec2ui(jj[0], ii[1])), // (i+1,j)
                          GetValueAt(vec2ui(ii[0], jj[1])), // (i,j+1)
                          GetValueAt(jj), // (i+1,j+1)
                          weight[0], weight[1]);
        }
    }
    
    
    __bidevice__ F DivergenceAtVertex(const U &index){
        F value(0);
        for(int i = 0; i < dimensions; i++){
            unsigned int idn = index[i] > 0 ? index[i]-1 : index[i];
            unsigned int idp = index[i] < resolution[i]-1 ? index[i]+1: index[i];
            U forward = index;
            U backward = index;
            forward[i] = idp;
            backward[i] = idn;
            
            F fValue = GetValueAt(forward);
            F bValue = GetValueAt(backward);
            
            value += 0.5 * (fValue - bValue) / spacing[i];
        }
        
        return value;
    }
    
    __bidevice__ T Gradient(const T &p){
        T value;
        // the way our setup works is we adjust resolution in Y/Z axis
        // so the reference spacing is X
        AssertA(!HasZero(spacing), "Zero spacing");
        Float d = spacing[0];
        Float inv = 1.0 / (2.0 * spacing[0]);
        
        // central differences at each axis
        for(int i = 0; i < dimensions; i++){
            T s(0); s[i] = d;
            Float forward  = Sample(p + s);
            Float backward = Sample(p - s);
            value[i] = (forward - backward) * inv;
        }
        
        return value;
    }
    
    __host__ void BuildFaceCentered(const F &initialValue = F(0)){
        T fres;
        perComponent[0] = 0, perComponent[1] = 0;
        perComponent[2] = 0;
        total = 0;
        filled = 0;
        for(int i = 0; i < dimensions; i++){
            fres[i] = (Float)resolution[i];
            int n = resolution[i]+1;
            for(int j = 0; j < dimensions; j++){
                if(i != j){
                    n *= resolution[j];
                }
            }
            
            perComponent[i] = n;
            total += n;
        }
        
        bounds = Q(minPoint, minPoint + spacing * fres);
        field = cudaAllocateVx(F, total);
        
        fieldUVW[0] = &field[0];
        fieldUVW[1] = nullptr; 
        fieldUVW[2] = nullptr;
        
        int at = perComponent[0];
        for(int i = 1; i < dimensions; i++){
            fieldUVW[i] = &field[at];
            at += perComponent[i];
        }
        
        int maxLen = Max(Max(perComponent[0], perComponent[2]), perComponent[1]);
        for(int i = 0; i < maxLen; i++){
            if(i < perComponent[0]) fieldUVW[0][i] = initialValue;
            if(i < perComponent[1]) fieldUVW[1][i] = initialValue;
            if(i < perComponent[2]) fieldUVW[2][i] = initialValue;
        }
    }
    
    __host__ void Build(const U &resol, const T &space, 
                        const T &origin, VertexType vtype, 
                        const F &initialValue = F(0))
    {
        T fres;
        SetDimension(T(0));
        resolution = resol;
        minPoint = origin;
        spacing = space;
        type = vtype;
        filled = 0;
        if(type == FaceCentered){
            BuildFaceCentered(initialValue);
        }else{
            total = 1;
            
            for(int i = 0; i < dimensions; i++){
                resolution[i] = Get1DLengthFor(resol[i]);
                total *= resolution[i];
                fres[i] = (Float)resol[i];
            }
            
            bounds = Q(origin, origin + spacing * fres);
            
            field = cudaAllocateVx(F, total);
            for(int i = 0; i < total; i++){
                field[i] = initialValue;
            }
        }
    }
};

typedef FieldGrid<vec2f, vec2ui, Bounds2f, Float> FieldGrid2f;
typedef FieldGrid<vec3f, vec3ui, Bounds3f, Float> FieldGrid3f;

template<typename T, typename U, typename Q>
class ContinuousParticleSetBuilder{
    public:
    std::vector<T> positions;
    std::vector<T> velocities;
    std::vector<T> forces;
    ParticleSet<T> *particleSet;
    Float kernelRadius;
    int maxNumOfParticles;
    bool warned;
    Grid<T, U, Q> *mappedDomain;
    std::set<unsigned int> mappedCellSet;
    std::map<unsigned int, std::vector<T>> mappedPositions;
    
    __host__ ContinuousParticleSetBuilder(int maxParticles=1000000){
        maxNumOfParticles = Max(1, maxParticles);
        particleSet = cudaAllocateVx(ParticleSet<T>, 1);
        particleSet->SetSize(maxNumOfParticles);
        int pSize = MaximumParticlesPerBucket;
        int *ids = cudaAllocateVx(int, pSize * maxParticles);
        int *ref = ids;
        for(int i = 0; i < maxParticles; i++){
            Bucket *bucket = particleSet->GetRawData(particleSet->buckets, i);
            bucket->SetPointer(&ref[i * pSize], pSize);
        }
        
        kernelRadius = 2.0 * 0.02; // TODO: default for now
        mappedDomain = nullptr;
        warned = false;
    }
    
    __host__ void SetKernelRadius(Float radius){
        kernelRadius = radius;
    }
    
    __host__ void MapGrid(Grid<T, U, Q> *grid){
        int count = particleSet->GetParticleCount();
        mappedDomain = grid;
        for(int i = 0; i < count; i++){
            T p = particleSet->GetParticlePosition(i);
            unsigned int h = grid->GetLinearHashedPosition(p);
            auto ret = mappedCellSet.insert(h);
            if(ret.first != mappedCellSet.end()){
                std::vector<T> pos;
                
                Cell<Q> *cell = grid->GetCell(h);
                ParticleChain *pChain = cell->GetChain();
                int size = cell->GetChainLength();
                AssureA(size > 0, "Called for MapGrid but grid is inconsistent, missing first distribution?");
                
                for(int j = 0; j < size; j++){
                    T pj = particleSet->GetParticlePosition(pChain->pId);
                    pos.push_back(pj);
                    pChain = pChain->next;
                }
                
                mappedPositions[h] = pos;
            }
        }
        
        std::set<unsigned int>::iterator it;
        for(it = mappedCellSet.begin(); it != mappedCellSet.end(); it++){
            unsigned int h = *it;
            std::vector<T> pos = mappedPositions[h];
            Cell<Q> *cell = mappedDomain->GetCell(h);
            AssureA(pos.size() == cell->GetChainLength(),
                    "Inconsistent grid mapping detected, invalid first distribution?");
        }
    }
    
    __host__ void MapGridEmit(const std::function<T(const T &)> velocity, Float d=0.02){
        std::set<unsigned int>::iterator it;
        int numNewParticles = 0;
        for(it = mappedCellSet.begin(); it != mappedCellSet.end(); it++){
            unsigned int h = *it;
            std::vector<T> pos = mappedPositions[h];
            Cell<Q> *cell = mappedDomain->GetCell(h);
            int size = cell->GetChainLength();
            if(size >= MaximumParticlesPerBucket) continue;
            int toInsert = Min(MaximumParticlesPerBucket - size, pos.size());

            for(int i = 0; i < toInsert; i++){
                T pi = pos[i];
                int can_add = 1;
                ParticleChain *pChain = cell->GetChain();
                for(int j = 0; j < size; j++){
                    T pj = particleSet->GetParticlePosition(pChain->pId);
                    if(Distance(pj, pi) < d){
                        can_add = 0;
                        break;
                    }
                    pChain = pChain->next;
                }
                
                if(can_add){
                    if(AddParticle(pi, velocity(pi))){
                        numNewParticles++;
                    }else{
                        Commit();
                        return;
                    }
                }
            }
        }
        
        if(numNewParticles > 0) Commit();
    }
    
    __host__ int AddParticle(const T &pos, const T &vel = T(0),
                             const T &force = T(0))
    {
        int total = particleSet->GetParticleCount() + positions.size();
        int ok = 0;
        if(total+1 <= maxNumOfParticles){
            positions.push_back(pos);
            velocities.push_back(vel);
            forces.push_back(force);
            ok = 1;
        }else if(!warned){
            printf("\nReached maximum builder capacity\n");
            warned = true;
        }
        
        return ok;
    }
    
    __host__ void Commit(){
        if(positions.size() > 0){
            int startId = particleSet->GetParticleCount();
            particleSet->AppendData(positions.data(), velocities.data(),
                                    forces.data(), positions.size());
            if(mappedDomain != nullptr){
                mappedDomain->DistributeByParticleList(particleSet, positions.data(), 
                                                       positions.size(), startId,
                                                       kernelRadius);
            }
            positions.clear();
            velocities.clear();
            forces.clear();
        }
    }
    
    __host__ ParticleSet<T> *GetParticleSet(){
        return particleSet;
    }
    
    __host__ int GetParticleCount(){
        return particleSet->GetParticleCount();
    }
};

typedef ContinuousParticleSetBuilder<vec2f, vec2ui, Bounds2f> ContinuousParticleSetBuilder2;
typedef ContinuousParticleSetBuilder<vec3f, vec3ui, Bounds3f> ContinuousParticleSetBuilder3;

__host__ SphParticleSet2 *SphParticleSet2FromContinuousBuilder(ContinuousParticleSetBuilder2 *builder);
__host__ SphParticleSet3 *SphParticleSet3FromContinuousBuilder(ContinuousParticleSetBuilder3 *builder);
