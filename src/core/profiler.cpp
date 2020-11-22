#include <map>
#include <cutil.h>
#include <statics.h>
#include <bits/stdc++.h>
#include <sstream>

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
    GPUTimer *gpuTimer;
    CPUTimer *cpuTimer;
    Execution *exec;
    
    __host__ Profiler(){ 
        exec = nullptr;
        cpuTimer = new CPUTimer;
        gpuTimer = new GPUTimer;
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
        for(auto &it : execMap){
            pairVector.push_back(it);
        }
        
        std::sort(pairVector.begin(), pairVector.end(), PairComparator);
        std::cout << "\nBubbles Profiler ======================= " << std::endl;
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
        
        std::cout << "========================================= " << std::endl;
    }
};

Profiler *profiler = nullptr;

void ProfilerPrepare(const char *funcname){
    std::string func(funcname);
    if(profiler == nullptr){
        profiler = new Profiler();
    }
    
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