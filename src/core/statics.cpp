#include <statics.h>

TimerList::TimerList(){
    gpuTimer = nullptr;
    cpuTimer = nullptr;
    active = 0;
}

int TimerList::Active(){
    return active;
}

void TimerList::Start(std::string event){
    if(!gpuTimer || !cpuTimer){
        GPUTimer *gTimer = new GPUTimer;
        CPUTimer *cTimer = new CPUTimer;
        gpuTimer = gTimer;
        cpuTimer = cTimer;
    }
    active = 1;
    headers.push_back(event);
    cpuTimer->Start();
    gpuTimer->Start();
}

void TimerList::StopAndNext(std::string event){
    gpuTimer->Stop();
    cpuTimer->Stop();
    headers.push_back(event);
    gpuElapsed.push_back(gpuTimer->TimeElapsed());
    cpuElapsed.push_back(cpuTimer->TimeElapsed());
    cpuTimer->Start();
    gpuTimer->Start();
    active = 1;
}

void TimerList::Stop(){
    if(active){
        gpuTimer->Stop();
        cpuTimer->Stop();
        gpuElapsed.push_back(gpuTimer->TimeElapsed());
        cpuElapsed.push_back(cpuTimer->TimeElapsed());
        active = 0;
    }
}

void TimerList::Reset(){
    if(active){
        gpuTimer->Stop();
        cpuTimer->Stop();
        active = 0;
    }

    gpuElapsed.clear();
    cpuElapsed.clear();
    headers.clear();
}

Float TimerList::GetElapsedGPU(int i){
    Float t = 0;
    if(i < gpuElapsed.size()){
        t = gpuElapsed.at(i);
    }

    return t;
}

Float TimerList::GetElapsedCPU(int i){
    Float t = 0;
    if(i < cpuElapsed.size()){
        t = cpuElapsed.at(i);
    }

    return t;
}

Float TimerList::GetTotalSummedTimeCPU(){
    Float t = 0;
    for(int i = 0; i < cpuElapsed.size(); i++)
        t += cpuElapsed.at(i);
    return t;
}

Float TimerList::GetTotalSummedTimeGPU(){
    Float t = 0;
    for(int i = 0; i < gpuElapsed.size(); i++)
        t += gpuElapsed.at(i);
    return t;
}

void TimerList::PrintEvents(){
    // TODO: we need a extra flag so we can tell if it is gpu event or cpu
    int n = gpuElapsed.size();
    std::cout << "Timer events ( " << GetTotalSummedTimeCPU() << " ms ):" << std::endl;
    for(int i = 0; i < n; i++){
        Float interval = GetElapsedCPU(i);
        std::cout << " - ";
        if(headers[i].size() == 0)
            std::cout << "(Unregistered): ";
        else
            std::cout << headers[i] << ": ";
        std::cout << interval << " ms" << std::endl;
    }
}


void LNMStats::Add(LNMData data){
    rawLNMData.push_back(data);
}

LNMData LNMStats::Average(LNMData *faster, LNMData *slower){
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
