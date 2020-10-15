#pragma once
#include <cutil.h>
#include <geometry.h>
#include <vector>

class CPUTimer{
    public:
    clock_t start, end;
    __host__ CPUTimer(){}
    __host__ void Start(){ start = clock(); }
    __host__ void Stop(){ end = clock(); }
    __host__ Float TimeElapsed() { return to_cpu_time(start, end) * 1000.0; }
    __host__ void Release(){}
};

class GPUTimer{
    public:
    cudaEvent_t start, end;
    __host__ GPUTimer(){ Setup(); }
    __host__ void Setup(){
        cudaEventCreate(&start);
        cudaEventCreate(&end);
    }
    __host__ void Start(){ cudaEventRecord(start); }
    __host__ void Stop(){
        cudaEventRecord(end);
        cudaEventSynchronize(end);
    }
    __host__ Float TimeElapsed(){
        Float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, end);
        return milliseconds;
    }
    
    __host__ void Release() {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
};

class TimerList{
    public:
    GPUTimer *gpuTimer;
    CPUTimer *cpuTimer;
    std::vector<Float> gpuElapsed;
    std::vector<Float> cpuElapsed;
    int active;
    __host__ TimerList();
    __host__ void Start();
    __host__ void Stop();
    __host__ void StopAndNext();
    __host__ void Reset();
    __host__ Float GetElapsedCPU(int i);
    __host__ Float GetElapsedGPU(int i);
};