#include <map>
#include <cutil.h>
#include <statics.h>
#include <bits/stdc++.h>
#include <sstream>

__global__ void GetKernelStats(int *buffer);

class Execution{
    public:
    long int runs;
    std::vector<double> gpuTime;
    std::vector<double> cpuTime;
    double cachedAverageCpu;
    double cachedAverageGpu;
    double minCpu;
    double minGpu;
    int has_cached;
    
    std::string callName;
    __host__ Execution(std::string name){
        callName = name;
        minCpu = Infinity;
        minGpu = Infinity;
        runs = 0;
    }
    
    __host__ void Increase(Float gpuDt, Float cpuDt){
        gpuTime.push_back(gpuDt);
        cpuTime.push_back(cpuDt);
        minGpu = minGpu > gpuDt ? gpuDt : minGpu;
        minCpu = minCpu > cpuDt ? cpuDt : minCpu;
        runs += 1;
        has_cached = 0;
    }
    
    __host__ long int GetAverage(double &gpu, double &cpu){
        if(!has_cached){
            double invRuns = 1.0 / (double)runs;
            gpu = 0;
            cpu = 0;
            for(long int i = 0; i < runs; i++){
                gpu += gpuTime[i] * invRuns;
                cpu += cpuTime[i] * invRuns;
            }
            
            cachedAverageCpu = cpu;
            cachedAverageGpu = gpu;
            has_cached = 1;
        }else{
            cpu = cachedAverageCpu;
            gpu = cachedAverageGpu;
        }
        
        return runs;
    }
    
    __host__ void GetMin(double &gpu, double &cpu){
        gpu = minGpu;
        cpu = minCpu;
    }
};

static int PairComparator(std::pair<std::string, Execution *> &a,
                          std::pair<std::string, Execution *> &b)
{
    Execution *ptrA = a.second;
    Execution *ptrB = b.second;
    double cpuA, gpuA, cpuB, gpuB;
    (void)ptrA->GetAverage(gpuA, cpuA);
    (void)ptrB->GetAverage(gpuB, cpuB);
    return gpuB < gpuA;
}

class Profiler{
    public:
    std::map<std::string, Execution *> execMap;
    int *particleInteraction;
    int particleCount;
    GPUTimer *gpuTimer;
    CPUTimer *cpuTimer;
    Execution *exec;
    
    __host__ Profiler(){ 
        exec = nullptr;
        cpuTimer = new CPUTimer;
        gpuTimer = new GPUTimer;
        particleCount = 0;
    }
    
    __host__ void AllocateKernelBuffers(int pCount){
        if(pCount > 0){
            particleCount = pCount;
            particleInteraction = cudaAllocateVx(int, pCount);
        }
    }
    
    __host__ void Prepare(const std::string &fname){
        if(execMap.find(fname) == execMap.end()){
            Execution *execution = new Execution(fname);
            execMap[fname] = execution;
        }
        
        exec = execMap[fname];
        gpuTimer->Start();
        cpuTimer->Start();
    }
    
    __host__ void Finish(){
        if(exec){
            gpuTimer->Stop();
            cpuTimer->Stop();
            Float cpums = cpuTimer->TimeElapsed();
            Float gpums = gpuTimer->TimeElapsed();
            //Float gpums = 0;
            exec->Increase(gpums, cpums);
            exec = nullptr;
        }
    }
    
    __host__ void Report(){
        std::vector<std::pair<std::string, Execution *>> pairVector;
        int maxIteractions = 0, minIteractions = 99999;
        double averageIteractions = 0;
        for(auto &it : execMap){
            pairVector.push_back(it);
        }
        
        std::sort(pairVector.begin(), pairVector.end(), PairComparator);
        if(particleCount > 0){
            double invP = 1.0 / (double)particleCount;
            GetKernelStats<<<1, 1>>>(particleInteraction);
            cudaDeviceSynchronize();
            for(int i = 0; i < particleCount; i++){
                int si = particleInteraction[i];
                maxIteractions = si > maxIteractions ? si : maxIteractions;
                minIteractions = si < minIteractions ? si : minIteractions;
                averageIteractions += ((double)si) * (invP);
            }
        }
        
        std::cout << "\nBubbles Profiler ======================= " << std::endl;
        std::cout << "Function executions" << std::endl;
        for(auto &it : pairVector){
            double cpu, gpu;
            std::stringstream ss;
            std::string name = it.first;
            Execution *e = it.second;
            (void)e->GetAverage(gpu, cpu);
            ss << "[ " << name << " ]  " << gpu << "ms";
            e->GetMin(gpu, cpu);
            ss << "  -- " << gpu << "ms";
            std::cout << ss.str() << std::endl;
        }
        
        if(particleCount > 0){
            std::cout << "Particle execution" << std::endl;
            std::cout << " Maximum iteraction " << maxIteractions << std::endl;
            std::cout << " Minimum iteraction " << minIteractions << std::endl;
            std::cout << " Average iteraction " << averageIteractions << std::endl;
        }
        
        std::cout << "========================================= " << std::endl;
    }
};

Profiler *profiler = nullptr;

class ProfilerKernel{
    public:
    int *particleDeviceBuffer;
    int size;
    
    __bidevice__ ProfilerKernel(){
        particleDeviceBuffer = nullptr;
        size = 0;
    }
    
    __bidevice__ void SetupParticleBuffer(int pCount){
        if(pCount > 0){
            if(particleDeviceBuffer) delete[] particleDeviceBuffer;
            particleDeviceBuffer = new int[pCount];
            size = pCount;
            memset(particleDeviceBuffer, 0x00, sizeof(int) * pCount);
        }
    }
    
    __bidevice__ void ReleaseParticleBuffer(){
        if(particleDeviceBuffer) delete[] particleDeviceBuffer;
        particleDeviceBuffer = nullptr;
        size = 0;
    }
    
    __bidevice__ int Size(){ return size; }
    
    __bidevice__ void Set(int value, int i){
        if(size > i){
            particleDeviceBuffer[i] = value;
        }
    }
};

__device__ ProfilerKernel *kernelProfiler = nullptr;

__global__ void InitializeKernelProfiler(int storage){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        kernelProfiler = new ProfilerKernel;
        kernelProfiler->SetupParticleBuffer(storage);
        printf("Kernel stats initialized for %d particles\n", storage);
    }
}

__global__ void ReleaseKernelProfiler(){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        kernelProfiler->ReleaseParticleBuffer();
        delete kernelProfiler;
    }
}

__global__ void GetKernelStats(int *buffer){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        if(kernelProfiler->Size() > 0){
            memcpy(buffer, kernelProfiler->particleDeviceBuffer,
                   kernelProfiler->Size() * sizeof(int));
        }
    }
}

__host__ __device__ 
void _ProfilerSetParticle(int value, int id){
#if defined(__CUDA_ARCH__)
    if(kernelProfiler)
        kernelProfiler->Set(value, id);
#else
    //TODO: CPU set
#endif
}

void ProfilerInitKernel(int pCount){
    (void)pCount;
#if defined(PROFILER_ALL)
    if(pCount > 0){
        InitializeKernelProfiler<<<1,1>>>(pCount);
        cudaDeviceSynchronize();
        profiler->AllocateKernelBuffers(pCount);
    }
#endif
}

void ProfilerInit(int pCount){
    if(profiler == nullptr){
        profiler = new Profiler();
        ProfilerInitKernel(pCount);
    }
}

void ProfilerSetParticleCount(int pCount){
    ProfilerInit(pCount);
}

void ProfilerPrepare(const char *funcname){
    std::string func(funcname);
    ProfilerInit(0);
    profiler->Prepare(func);
}

void ProfilerFinish(){
    if(profiler)
        profiler->Finish();
}

void ProfilerReport(){
    if(profiler)
        profiler->Report();
}