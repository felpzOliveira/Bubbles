#if !defined(CUTIL_H)
#define CUTIL_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <iostream>
#include <map>
#include <typeindex>
#include <profiler.h>

/*
* CUDA UTILITIES
*/
typedef enum{
    MaxOccupancyBlockSize=0, CustomizedBlockSize
}CudaLaunchStrategy;

typedef enum{
    GPU=0, CPU
}AllocatorType;

#define CUCHECK(r) _check((r), __LINE__, __FILE__)
#define cudaAllocate(bytes) _cudaAllocate(bytes, __LINE__, __FILE__, true)
#define cudaAllocateEx(bytes, abort) _cudaAllocate(bytes, __LINE__, __FILE__, abort)
#define cudaAllocateVx(type, n) (type *)_cudaAllocate(sizeof(type)*n, __LINE__, __FILE__, true)
#define cudaDeviceAssert(fname) if(cudaKernelSynchronize()){ printf("Failure for %s\n", fname); cudaSafeExit(); }

#define __bidevice__ __host__ __device__ 
#define MAX(a, b) a > b ? a : b

#define GPU_LAMBDA(...) [=] __device__(__VA_ARGS__)

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
void _check(cudaError_t err, int line, const char *filename);

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


template<typename T>
class DataBuffer{
    public:
    T *data;
    int size;
    
    __bidevice__ DataBuffer(){ size = 0; data = nullptr; }
    
    __bidevice__ int GetSize(){
        return size;
    }
    
    __host__ void SetSize(int n){
        size = n;
        data = cudaAllocateVx(T, size);
    }
    
    __host__ int SetDataAt(T *values, int n, int at){
        int rv = 1;
        if(size >= at + n){
            memcpy(&data[at], values, n * sizeof(T));
            rv = 0;
        }else{
            printf("Warning: Invalid fill index {%d + %d >= %d}\n", at, n, size);
        }
        
        return rv;
    }
    
    __host__ void Clear(){
        memset(data, 0x0, sizeof(T) * size);
    }
    
    __host__ void SetData(T *values, int n){
        size = n;
        data = cudaAllocateVx(T, size);
        memcpy(data, values, sizeof(T) * size);
    }
    
    __bidevice__ T At(int i){
        if(i < size) return data[i];
        printf("Warning: Invalid query index {%d >= %d}\n", i, size);
        return T(0);
    }
    
    __bidevice__ T *Get(int i){
        if(i < size) return &data[i];
        printf("Warning: Invalid get index {%d >= %d}\n", i, size);
        return nullptr;
    }
    
    __bidevice__ void Set(T val, int i){
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

template<typename F> __global__ void GenericKernel(F fn, int items){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= items) return;

    fn(tid);
}

template<typename F> inline
__host__ void GPUParallelLambda(const char *desc, int nItems, F fn){
    auto kernel = &GenericKernel<F>;
    int blockSize = GetBlockSize(kernel, desc);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    ProfilerPrepare(desc);
    kernel<<<gridSize, blockSize>>>(fn, nItems);
    cudaDeviceAssert(desc);
    ProfilerFinish();
}

#define GPUKernel(...) (__VA_ARGS__)
#define GPULaunchItens(nItems, __blockSize, call, ...)\
{\
    int __gridSize = (nItems + __blockSize - 1) / __blockSize;\
    ProfilerPrepare(#call);\
    call<<<__gridSize, __blockSize>>>(__VA_ARGS__);\
    cudaDeviceAssert(#call);\
    ProfilerFinish();\
}

#define GPULaunch(nItems, call, ...)\
{\
    int __blockSize = GetBlockSize(call, #call);\
    GPULaunchItens(nItems, __blockSize, call, __VA_ARGS__);\
}

#endif
