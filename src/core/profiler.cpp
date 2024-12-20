#include <map>
#include <cutil.h>
#include <statics.h>
#include <bits/stdc++.h>
#include <sstream>

bb_kernel void GetKernelStats(int *buffer);

class Execution{
    public:
    long int runs;
    std::vector<double> gpuTime;
    std::vector<double> cpuTime;
    CPUTimer *cpuTimer;
    double cachedAverageCpu;
    double cachedAverageGpu;
    double minCpu;
    double minGpu;
    int has_cached;

    std::string callName;
    Execution(std::string name){
        callName = name;
        minCpu = Infinity;
        minGpu = Infinity;
        cpuTimer = new CPUTimer;
        runs = 0;
    }

    void Increase(Float gpuDt, Float cpuDt){
        gpuTime.push_back(gpuDt);
        cpuTime.push_back(cpuDt);
        minGpu = minGpu > gpuDt ? gpuDt : minGpu;
        minCpu = minCpu > cpuDt ? cpuDt : minCpu;
        runs += 1;
        has_cached = 0;
    }

    void GetLatest(double &gpu, double &cpu){
        gpu = 0;
        cpu = 0;
        if(runs > 0){
            gpu = gpuTime[runs-1];
            cpu = cpuTime[runs-1];
        }
    }

    long int GetAverage(double &gpu, double &cpu){
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

    void GetMin(double &gpu, double &cpu){
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

static Float stpInterval = 0;
static int stpIterations = 0;
static GPUTimer *stpTimer = nullptr;

class Profiler{
    public:
    std::map<std::string, Execution *> execMap;
    std::stack<Execution *> execStack;
    int *particleInteraction;
    int particleCount;
    GPUTimer *gpuTimer;
    CPUTimer *cpuTimer;
    Execution *exec;
    Execution *manualExec;
    Float fastestStep;

    Profiler(){
        exec = nullptr;
        manualExec = nullptr;
        cpuTimer = new CPUTimer;
        gpuTimer = new GPUTimer;
        particleCount = 0;
        fastestStep = FLT_MAX;
    }

    void AllocateKernelBuffers(int pCount){
        if(pCount > 0){
            particleCount = pCount;
            particleInteraction = cudaAllocateVx(int, pCount);
        }
    }

    double GetResult(const std::string &fname){
        double es = -1, ev = -1;
        if(execMap.find(fname) != execMap.end()){
            Execution *e = execMap[fname];
            e->GetLatest(es, ev);
        }

        return ev;
    }

    void ManualPrepare(const std::string &fname){
        if(execMap.find(fname) == execMap.end()){
            Execution *execution = new Execution(fname);
            execMap[fname] = execution;
        }

        manualExec = execMap[fname];
        execStack.push(manualExec);
        manualExec->cpuTimer->Start();
    }

    void ManualFinish(){
        if(manualExec){
            manualExec->cpuTimer->Stop();
            Float cpums = manualExec->cpuTimer->TimeElapsed();
            manualExec->Increase(cpums, cpums);
            if(execStack.size() > 0){
                manualExec = execStack.top();
                execStack.pop();
            }else{
                manualExec = nullptr;
            }
        }
    }

    void Prepare(const std::string &fname){
        if(execMap.find(fname) == execMap.end()){
            Execution *execution = new Execution(fname);
            execMap[fname] = execution;
        }

        exec = execMap[fname];
        gpuTimer->Start();
        cpuTimer->Start();
    }

    void Finish(){
        if(exec){
            gpuTimer->Stop();
            cpuTimer->Stop();
            Float cpums = cpuTimer->TimeElapsed();
            Float gpums = gpuTimer->TimeElapsed();
            exec->Increase(gpums, cpums);
            exec = nullptr;
        }
    }

    void Report(int frameId){
        std::vector<std::pair<std::string, Execution *>> pairVector;
        int maxIteractions = 0, minIteractions = 99999;
        double averageIteractions = 0;
        double estimatedIncrease = 0;
        double diffCpuTime = 0;
        double estimatedGpuTime = 0;
        int iterationsSteps = ProfilerGetIterationsPassed();
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

        fastestStep = fastestStep > stpInterval ? stpInterval : fastestStep;
        diffCpuTime = stpInterval - fastestStep;
        std::cout << "\nBubbles Profiler ============================================== "
            << std::endl;
        std::cout << "Step interval: " << stpInterval << " ms [ " <<
            iterationsSteps << " ]";
        if(frameId < 0)
            std::cout << std::endl;
        else
            std::cout << " [ " << frameId << " ] " << std::endl;
        std::cout << "Function executions" << std::endl;
        for(auto &it : pairVector){
            double cpu, gpu;
            double mgpu = 0;
            double di = 0;
            std::stringstream ss;
            std::string name = it.first;
            Execution *e = it.second;
            (void)e->GetLatest(mgpu, cpu);
            estimatedGpuTime += mgpu;

            ss << "[ " << name << " ]  " << mgpu << "ms";
            e->GetMin(gpu, cpu);
            ss << " -- " << gpu << "ms";
            di = mgpu - gpu;
            ss << " -- " << di << "ms";
            estimatedIncrease += di;
            std::cout << ss.str() << std::endl;
        }

        std::cout << "Estimated increase  [ GPU ] = " <<
            estimatedIncrease << "ms" << std::endl;

        std::cout << "Real clock increase [ CPU ] = " <<
            diffCpuTime << "ms" << std::endl;

        std::cout << "Estimated execution [ GPU ] = " <<
            estimatedGpuTime << "ms" << std::endl;

        if(particleCount > 0){
            std::cout << "Particle execution [ " << particleCount << " ]" <<  std::endl;
            std::cout << " Maximum iteraction " << maxIteractions << std::endl;
            std::cout << " Minimum iteraction " << minIteractions << std::endl;
            std::cout << " Average iteraction " << averageIteractions << std::endl;
        }

        std::cout << "================================================================ "
            << std::endl;
    }
};

Profiler *profiler = nullptr;

class ProfilerKernel{
    public:
    int *particleDeviceBuffer;
    int size;

    bb_cpu_gpu ProfilerKernel(){
        particleDeviceBuffer = nullptr;
        size = 0;
    }

    bb_cpu_gpu void SetupParticleBuffer(int pCount){
        if(pCount > 0){
            if(particleDeviceBuffer) delete[] particleDeviceBuffer;
            particleDeviceBuffer = new int[pCount];
            size = pCount;
            memset(particleDeviceBuffer, 0x00, sizeof(int) * pCount);
        }
    }

    bb_cpu_gpu void ReleaseParticleBuffer(){
        if(particleDeviceBuffer) delete[] particleDeviceBuffer;
        particleDeviceBuffer = nullptr;
        size = 0;
    }

    bb_cpu_gpu int Size(){ return size; }

    bb_cpu_gpu void Set(int value, int i){
        if(size > i){
            particleDeviceBuffer[i] = value;
        }
    }
};

__device__ ProfilerKernel *kernelProfiler = nullptr;

bb_kernel void InitializeKernelProfiler(int storage){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        kernelProfiler = new ProfilerKernel;
        kernelProfiler->SetupParticleBuffer(storage);
        printf("Kernel stats initialized for %d particles\n", storage);
    }
}

bb_kernel void ReleaseKernelProfiler(){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        kernelProfiler->ReleaseParticleBuffer();
        delete kernelProfiler;
    }
}

bb_kernel void GetKernelStats(int *buffer){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        if(kernelProfiler->Size() > 0){
            memcpy(buffer, kernelProfiler->particleDeviceBuffer,
                   kernelProfiler->Size() * sizeof(int));
        }
    }
}

bb_cpu_gpu
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

void ProfilerBeginStep(){
    if(stpTimer == nullptr){
        stpTimer = new GPUTimer;
        stpInterval = 0;
    }
    stpIterations = 0;
    stpTimer->Start();
}

void ProfilerIncreaseStepIteration(){
    stpIterations += 1;
}

void ProfilerEndStep(){
    stpTimer->Stop();
    stpInterval = stpTimer->TimeElapsed();
}

double ProfilerGetStepInterval(){
    return (double)stpInterval;
}

int ProfilerGetIterationsPassed(){
    return stpIterations;
}

void ProfilerManualStart(const char *funcname){
    if(profiler)
        profiler->ManualPrepare(funcname);
}

void ProfilerManualFinish(){
    if(profiler)
        profiler->ManualFinish();
}

double ProfilerGetEvaluation(const char *fname){
    double e = 0;
    if(profiler)
        e = profiler->GetResult(fname);
    return e;
}

void ProfilerReport(int frameId){
    if(profiler)
        profiler->Report(frameId);
}
