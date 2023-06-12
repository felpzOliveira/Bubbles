#pragma once
#include <cutil.h>
#include <geometry.h>
#include <vector>

class CPUTimer{
    public:
    clock_t start, end;
    CPUTimer(){}
    void Start(){ start = clock(); }
    void Stop(){ end = clock(); }
    Float TimeElapsed() { return to_cpu_time(start, end) * 1000.0; }
    void Release(){}
};

class GPUTimer{
    public:
    cudaEvent_t start, end;
    GPUTimer(){ Setup(); }
    void Setup(){
        cudaEventCreate(&start);
        cudaEventCreate(&end);
    }
    void Start(){ cudaEventRecord(start); }
    void Stop(){
        cudaEventRecord(end);
        cudaEventSynchronize(end);
    }
    float TimeElapsed(){
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, end);
        return milliseconds;
    }

    void Release() {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
};

class LNMData{
    public:
    Float timeTaken;
    Float simPercentage;
    bb_cpu_gpu LNMData(): timeTaken(0), simPercentage(0){}
    bb_cpu_gpu LNMData(Float ttaken, Float percentage) : timeTaken(ttaken),
    simPercentage(percentage){}
};

class LNMStats{
    public:
    std::vector<LNMData> rawLNMData;
    LNMStats(){}
    void Add(LNMData data);
    LNMData Average(LNMData *faster=nullptr, LNMData *slower=nullptr);
    LNMData Last(){
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
    std::vector<std::string> headers;
    std::vector<Float> gpuElapsed;
    std::vector<Float> cpuElapsed;
    int active;
    TimerList();
    void Start(std::string event=std::string());
    void Stop();
    void StopAndNext(std::string event=std::string());
    void Reset();
    void PrintEvents();
    int Active();
    Float GetElapsedCPU(int i);
    Float GetElapsedGPU(int i);
    Float GetTotalSummedTimeCPU();
    Float GetTotalSummedTimeGPU();
};
