#include <statics.h>

__host__ TimerList::TimerList(){
    gpuTimer = nullptr;
    cpuTimer = nullptr;
    active = 0;
}

__host__ void TimerList::Start(){
    if(!gpuTimer || !cpuTimer){
        GPUTimer *gTimer = new GPUTimer;
        CPUTimer *cTimer = new CPUTimer;
        gpuTimer = gTimer;
        cpuTimer = cTimer;
    }
    active = 1;
    cpuTimer->Start();
    gpuTimer->Start();
}

__host__ void TimerList::StopAndNext(){
    gpuTimer->Stop();
    cpuTimer->Stop();
    gpuElapsed.push_back(gpuTimer->TimeElapsed());
    cpuElapsed.push_back(cpuTimer->TimeElapsed());
    cpuTimer->Start();
    gpuTimer->Start();
    active = 1;
}

__host__ void TimerList::Stop(){
    if(active){
        gpuTimer->Stop();
        cpuTimer->Stop();
        gpuElapsed.push_back(gpuTimer->TimeElapsed());
        cpuElapsed.push_back(cpuTimer->TimeElapsed());
        active = 0;
    }
}

__host__ void TimerList::Reset(){
    if(active){
        gpuTimer->Stop();
        cpuTimer->Stop();
        active = 0;
    }
    
    gpuElapsed.clear();
    cpuElapsed.clear();
}

__host__ Float TimerList::GetElapsedGPU(int i){
    Float t = 0;
    if(i < gpuElapsed.size()){
        t = gpuElapsed.at(i);
    }
    
    return t;
}

__host__ Float TimerList::GetElapsedCPU(int i){
    Float t = 0;
    if(i < cpuElapsed.size()){
        t = cpuElapsed.at(i);
    }
    
    return t;
}