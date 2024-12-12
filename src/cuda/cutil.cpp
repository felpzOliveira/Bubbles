#include <cutil.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <sstream>

//#define PRINT_INIT

Memory global_memory = {0};
CudaExecutionStrategy global_cuda_strategy = {
    .strategy = CudaLaunchStrategy::MaxOccupancyBlockSize,
    .blockSize = 0,
};

void _check(cudaError_t err, const char *cmd, int line, const char *filename){
    if(err != cudaSuccess){
        std::cout << "Aborting ==============" << std::endl;
        std::cout << "  CUDA error: \'" << cmd << "\'" << std::endl;
        std::cout << "  Location: " << filename << ":" << line << std::endl;
        std::cout << "  Reason: " << cudaGetErrorString(err) << " [ " << err << " ]" << std::endl;
        std::cout << "=======================" << std::endl;
        cudaDeviceReset();
        getchar();
        exit(0);
    }
}

std::string get_time_string(clock_t start, clock_t end, int i, int it){
    double tt = to_cpu_time(start, end);
    double est = tt * it / (i + 1);
    std::stringstream stt, sest;
    stt << tt; sest << est;
    std::string ts = time_to_string(stt.str(), 8);
    std::string vs = time_to_string(sest.str(), 8);
    std::string resp("( ");
    resp += ts; resp += "s | ";
    resp += vs; resp += "s )";
    return resp;
}

std::string time_to_string(std::string val, int size){
    std::string resp(val);
    int dif = size - val.size();
    if(dif < 0){ // truncate
        resp = val.substr(0, size);
    }else if(dif > 0){
        int has_dot = 0;
        for(int i = 0; i < val.size(); i++){
            if(val[i] == '.'){
                has_dot = 1;
                break;
            }
        }

        if(has_dot){
            for(int i = 0; i < dif; i++) resp += "0";
        }else{
            std::string out;
            for(int i = 0; i < dif; i++) out += "0";
            out += val;
            resp = out;
        }
    }

    return resp;
}

double to_cpu_time(clock_t start, clock_t end){
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

std::string get_time_unit(double *inval){
    std::string unit("s");
    double val = *inval;
    if(val > 60){
        unit = "min";
        val /= 60.0;
    }

    if(val > 60){
        unit = "h";
        val /= 60;
    }

    *inval = val;
    return unit;
}

void cudaInitEx(){
    (void)cudaInit();
}

int cudaInit(){
    int nDevices;
    int dev;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
#if defined(PRINT_INIT)
        printf(" > Device name: %s\n", prop.name);
        printf(" > Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf(" > Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf(" > Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
#endif
    }

    if(nDevices > 0){
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(cudaDeviceProp));
        prop.major = 1; prop.minor = 0;
        CUCHECK(cudaChooseDevice(&dev, &prop));
        CUCHECK(cudaGetDeviceProperties(&prop, dev));
        global_memory.allocated = 0;
#if defined(PRINT_INIT)
        std::cout << "Using device " << prop.name << " [ " <<  prop.major << "." << prop.minor << " ]" << std::endl;
#endif
        clock_t start = clock();
        cudaFree(0);
        clock_t mid = clock();

        cudaDeviceReset();
        clock_t end = clock();

        double cpu_time_mid = to_cpu_time(start, mid);
        double cpu_time_reset = to_cpu_time(mid, end);
        double cpu_time_end = to_cpu_time(start, end);

        std::string unitAlloc = get_time_unit(&cpu_time_mid);
        std::string unitReset = get_time_unit(&cpu_time_reset);
        std::string unitTotal = get_time_unit(&cpu_time_end);

        std::string state("[OK]");
        if(cpu_time_end > 1.5){
            state = "[SLOW]";
        }
#if defined(PRINT_INIT)
        std::cout << "GPU init stats " << state << "\n" <<
            " > Allocation: " << cpu_time_mid << " " << unitAlloc << std::endl;
        std::cout << " > Reset: " << cpu_time_reset << " " << unitReset << std::endl;
        std::cout << " > Global: " << cpu_time_end << " " << unitTotal << std::endl;
#endif

    }

    return dev;
}

void cudaPrintMemoryTaken(){
    std::string unity("b");
    float amount = (float)(global_memory.allocated);
    if(amount > 1024){
        amount /= 1024.f;
        unity = "KB";
    }

    if(amount > 1024){
        amount /= 1024.f;
        unity = "MB";
    }

    if(amount > 1024){
        amount /= 1024.f;
        unity = "GB";
    }

    std::cout << "Took " << amount << " " << unity << " of GPU memory" << std::endl;
}

int cudaKernelSynchronize(){
    int rv = 0;
    cudaError_t errAsync = cudaDeviceSynchronize();
    cudaError_t errSync = cudaGetLastError();
    if(errSync != cudaSuccess){
        std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
        rv = 1;
    }
    if(errAsync != cudaSuccess){
        std::cout << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
        rv = 1;
    }

    return rv;
}

DeviceMemoryStats cudaReportMemoryUsage(){
    DeviceMemoryStats memStats;
    cudaError_t status = cudaMemGetInfo(&memStats.free_bytes, &memStats.total_bytes);
    if(status != cudaSuccess){
        std::cout << "Could not query device for memory!" << std::endl;
        memStats.valid = 0;
    }else{
        memStats.used_bytes = memStats.total_bytes - memStats.free_bytes;
        memStats.valid = 1;
    }

    return memStats;
}

void cudaSetLaunchStrategy(CudaLaunchStrategy strategy, int blockSize){
    global_cuda_strategy.strategy = strategy;
    global_cuda_strategy.blockSize = blockSize;
}

int cudaHasMemory(size_t bytes){
    DeviceMemoryStats mem = cudaReportMemoryUsage();
    int ok = 0;
    if(mem.valid){
        ok = mem.free_bytes > bytes ? 1 : 0;
    }

    return ok;
}

void cudaSafeExit(){
    cudaDeviceReset();
    exit(0);
}

int kUseThreadsNum = -1;
int kUseCPU = 0;
int GetConfiguredCPUThreads(){
    return kUseThreadsNum;
}

int GetSystemUseCPU(){
    return kUseCPU;
}

void SetSystemUseCPU(){
    kUseCPU = 1;
}

void SetSystemUseGPU(){
    kUseCPU = 0;
}

void SetCPUThreads(int nThreads){
    kUseThreadsNum = MAX(1, nThreads);
}
