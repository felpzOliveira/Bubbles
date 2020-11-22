#include <statics.h>

__host__ TimerList::TimerList(){
    gpuTimer = nullptr;
    cpuTimer = nullptr;
    active = 0;
}

__host__ int TimerList::Active(){
    return active;
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

__host__ void LNMStats::Add(LNMData data){
    rawLNMData.push_back(data);
}

__host__ LNMData LNMStats::Average(LNMData *faster, LNMData *slower){
    LNMData res;
    Float fastTime = FLT_MAX;
    Float slowTime = -FLT_MAX;
    int fastId = 0;
    int slowId = 0;
    int it = 0;
    
    res.timeTaken = 0;
    res.simPercentage = 0;
    if(rawLNMData.size() > 0){
        for(LNMData &data : rawLNMData){
            if(data.timeTaken > slowTime){
                slowTime = data.timeTaken;
                slowId = it;
            }
            
            if(data.timeTaken < fastTime){
                fastTime = data.timeTaken;
                fastId = it;
            }
            
            res.timeTaken += data.timeTaken;
            res.simPercentage += data.simPercentage;
            it++;
        }
        
        if(faster){
            *faster = rawLNMData[fastId];
        }
        
        if(slower){
            *slower = rawLNMData[slowId];
        }
        
        res.timeTaken /= (Float)it;
        res.simPercentage /= (Float)it;
    }
    
    return res;
}