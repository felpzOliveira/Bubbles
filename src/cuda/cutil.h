#if !defined(CUTIL_H)
#define CUTIL_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <iostream>
#include <map>
#include <typeindex>
#include <profiler.h>
#include <thread>
#include <vector>

/*
* CUDA UTILITIES
*/
typedef enum{
    MaxOccupancyBlockSize=0, CustomizedBlockSize
}CudaLaunchStrategy;

typedef enum{
    GPU=0, CPU
}AllocatorType;

#define CUCHECK(r) _check((r), #r, __LINE__, __FILE__)
#define cudaAllocate(bytes) _cudaAllocate(bytes, __LINE__, __FILE__, true)
#define cudaAllocateExclusive(bytes) _cudaAllocateExclusive(bytes, __LINE__, __FILE__, true)
#define cudaAllocateEx(bytes, abort) _cudaAllocate(bytes, __LINE__, __FILE__, abort)
#define cudaAllocateVx(type, n) (type *)_cudaAllocate(sizeof(type)*n, __LINE__, __FILE__, true)
#define cudaDeviceAssert(fname) if(cudaKernelSynchronize()){ printf("Failure for %s\n", fname); cudaSafeExit(); }
#define cudaAllocateUnregisterVx(type, n) (type *)_cudaAllocateUnregister(sizeof(type)*n, __LINE__, __FILE__, true)

#define bb_kernel __global__
#define bb_cpu __host__
#define bb_gpu __device__
#define bb_cpu_gpu __host__ __device__
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define GPU_LAMBDA(...) [=] bb_gpu(__VA_ARGS__)

typedef struct{
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
    int valid;
}DeviceMemoryStats;

typedef struct{
    size_t allocated;
}Memory;

typedef struct{
    CudaLaunchStrategy strategy;
    int blockSize;
}CudaExecutionStrategy;

extern Memory global_memory;
extern CudaExecutionStrategy global_cuda_strategy;

/*
* Sanity function to check for cuda* operations.
*/
void _check(cudaError_t err, const char *cmd, int line, const char *filename);

/*
* Initialize a cuda capable device to start kernel launches.
*/
int  cudaInit(void);
void cudaInitEx(void);

/*
* Synchronizes the device so host access is not asynchronous,
* also checks devices for errors in recent kernel launches.
*/
int cudaKernelSynchronize(void);

/*
* Get information about memory usage from the device.
*/
DeviceMemoryStats cudaReportMemoryUsage(void);

/*
* Checks if _at_this_moment_ it is possible to alocate memory on the device.
*/
int cudaHasMemory(size_t bytes);

/*
* Prints current amount of allocated device memory.
*/
void cudaPrintMemoryTaken(void);

/*
* Invokes the cuda reset operation for releasing memory and terminates
* software execution.
*/
void cudaSafeExit(void);

/*
* Sets the cuda strategy for kernel launch.
*/
void cudaSetLaunchStrategy(CudaLaunchStrategy strategy, int blockSize=0);

/*
* Returns the time passed between 'start' and 'end' in seconds.
*/
double to_cpu_time(clock_t start, clock_t end);

/*
* Returns a string with time difference and a estimation of the time
* left considering iteration 'i' and amount of iterations 'it'.
*/
std::string get_time_string(clock_t start, clock_t end, int i, int it);

/*
* Returns a string containing a time unit for the given time interval.
* Interval is adjusted for the unit representation.
*/
std::string get_time_unit(double *inval);

/*
* Fixes the string given to the size 'size' filling appropriates zeros
* for the given time.
*/
std::string time_to_string(std::string val, int size);

/*
* Attempts to allocate a block of memory in the device. The returned
* memory when valid (!=nullptr) is managed. This function register the 
* address in the active memory region implemented by memory.h.
* NOTE: Do *not* call cudaFree in the pointers returned, for freeing memory
* use either CudaMemoryManagerClearCurrent or CudaMemoryManagerClearAll
* after a region is no longer nedded.
*/
void *_cudaAllocate(size_t bytes, int line, const char *filename, bool abort);
void *_cudaAllocateUnregister(size_t bytes, int line, const char *filename, bool abort);
void *_cudaAllocateExclusive(size_t bytes, int line, const char *filename, bool abort);

template<typename T>
class DataBuffer{
    public:
    T *data;
    int size;

    bb_cpu_gpu DataBuffer(){ size = 0; data = nullptr; }

    bb_cpu_gpu int GetSize(){
        return size;
    }

    void SetSize(int n){
        size = n;
        data = cudaAllocateVx(T, size);
    }

    int SetDataAt(T *values, int n, int at){
        int rv = 1;
        if(size >= at + n){
            memcpy(&data[at], values, n * sizeof(T));
            rv = 0;
        }else{
            printf("Warning: Invalid fill index {%d + %d >= %d}\n", at, n, size);
        }

        return rv;
    }

    void Clear(){
        memset(data, 0x0, sizeof(T) * size);
    }

    void SetData(T *values, int n){
        size = n;
        data = cudaAllocateVx(T, size);
        memcpy(data, values, sizeof(T) * size);
    }

    bb_cpu_gpu T At(int i){
        if(i < size) return data[i];
        printf("Warning: Invalid query index {%d >= %d}\n", i, size);
        return T(0);
    }

    bb_cpu_gpu T *Get(int i){
        if(i < size) return &data[i];
        printf("Warning: Invalid get index {%d >= %d}\n", i, size);
        return nullptr;
    }

    bb_cpu_gpu void Set(T val, int i){
        if(i < size) data[i] = val;
        else printf("Warning: Invalid set index {%d >= %d}\n", i, size);
    }
};

template<typename F>
inline int GetBlockSize(F kernel, const char *fname){
    static std::map<std::type_index, int> kernelBlockSizes;
    std::type_index index = std::type_index(typeid(F));
    int blockSize = 0;

    if(global_cuda_strategy.strategy == CudaLaunchStrategy::CustomizedBlockSize){
        blockSize = global_cuda_strategy.blockSize;
    }else if(global_cuda_strategy.strategy == CudaLaunchStrategy::MaxOccupancyBlockSize){
        auto iter = kernelBlockSizes.find(index);
        if(iter != kernelBlockSizes.end()){
            blockSize = iter->second;
        }else{
            int minGridSize;
            CUCHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                                       kernel, 0, 0));
            kernelBlockSizes[index] = blockSize;
        }
    }
    return blockSize;
}

template<typename F> bb_kernel void GenericKernel(F fn, int items){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= items) return;

    fn(tid);
}

template<typename F> inline
void GPUParallelLambda(const char *desc, int nItems, F fn){
    auto kernel = &GenericKernel<F>;
    int blockSize = GetBlockSize(kernel, desc);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    ProfilerPrepare(desc);
    kernel<<<gridSize, blockSize>>>(fn, nItems);
    cudaDeviceAssert(desc);
    cudaDeviceSynchronize();
    ProfilerFinish();
}

#define GPUKernel(...) (__VA_ARGS__)
#define GPULaunchItens(nItems, __blockSize, call, ...)\
{\
    int __gridSize = (nItems + __blockSize - 1) / __blockSize;\
    ProfilerPrepare(#call);\
    call<<<__gridSize, __blockSize>>>(__VA_ARGS__);\
    cudaDeviceSynchronize();\
    ProfilerFinish();\
}

#define GPULaunch(nItems, call, ...)\
{\
    int __blockSize = GetBlockSize(call, #call);\
    GPULaunchItens(nItems, __blockSize, call, __VA_ARGS__);\
}


// CPU

/*
 * Returns the amount of cores in as seen by the system or 1 in case
 * a failure happens.
 */
inline int GetConcurrency(){
    return MAX(1, (int)std::thread::hardware_concurrency());
}

/*
 * Get user configuration of number of threads to use.
 */
int GetConfiguredCPUThreads();

/*
 * Configures how many threads CPU should use.
 */
void SetCPUThreads(int nThreads);

/*
 * Configures the system to use CPU parallelism instead of GPU.
 */
void SetSystemUseCPU();

/*
 * Configures the system to use GPU parallism.
 */
void SetSystemUseGPU();

/*
 * Check which device should use for parallelism.
 */
int GetSystemUseCPU();

/*
 * CPU parallel for. Divides the range [start, end) into slices and spawn
 * either 'kUseThreadsNum' or the system core count threads to solve the slices
 * using the function 'fn'. Setting 'kUseThreadsNum' to 1 makes this function
 * perform a simple for loop for easier debug.
 */
template<typename Index, typename Function>
void ParallelFor(Index start, Index end, Function fn){
    if(start > end) return;
    int userThreads = GetConfiguredCPUThreads();

    if(userThreads == 1){
        for(Index j = start; j < end; j++){
            fn(j);
        }
    }else{
        std::vector<std::thread> threads;
        int numThreads = GetConcurrency();

        numThreads = userThreads < 0 ? numThreads : MIN(userThreads, numThreads);

        Index n = end - start + 1;
        Index slice = (Index)std::round(n / (double)numThreads);
        slice = MAX(slice, Index(1));

        auto helper = [&fn](Index j1, Index j2){
            for(Index j = j1; j < j2; j++){
                fn(j);
            }
        };

        threads.reserve(numThreads);
        Index i0 = start;
        Index i1 = MIN(start + slice, end);
        for(int i = 0; i + 1 < numThreads && i0 < end; i++){
            threads.emplace_back(helper, i0, i1);
            i0 = i1;
            i1 = MIN(i1 + slice, end);
        }

        if(i0 < end){
            threads.emplace_back(helper, i0, end);
        }

        for(std::thread &th : threads){
            if(th.joinable()){
                th.join();
            }
        }
    }
}

/* Macro for auto-detection and parallelization */
#define AutoLambda(...) GPU_LAMBDA(__VA_ARGS__)

template<typename Index, typename Function>
void AutoParallelFor(const char *title, Index items, Function fn){
    if(GetSystemUseCPU()){
        ParallelFor((size_t)0, (size_t)items, fn);
    }else{
        GPUParallelLambda(title, items, fn);
    }
}

#endif
