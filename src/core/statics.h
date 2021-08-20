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
    __host__ float TimeElapsed(){
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, end);
        return milliseconds;
    }
    
    __host__ void Release() {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
};

class LNMData{
    public:
    Float timeTaken;
    Float simPercentage;
    __bidevice__ LNMData(): timeTaken(0), simPercentage(0){}
    __bidevice__ LNMData(Float ttaken, Float percentage) : timeTaken(ttaken), 
    simPercentage(percentage){}
};

class LNMStats{
    public:
    std::vector<LNMData> rawLNMData;
    __host__ LNMStats(){}
    __host__ void Add(LNMData data);
    __host__ LNMData Average(LNMData *faster=nullptr, LNMData *slower=nullptr);
    __host__ LNMData Last(){
        LNMData data(0,0);
        if(rawLNMData.size() > 0){
            data = rawLNMData[rawLNMData.size()-1];
        }
        
        return data;
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
    __host__ int Active();
    __host__ Float GetElapsedCPU(int i);
    __host__ Float GetElapsedGPU(int i);
};
