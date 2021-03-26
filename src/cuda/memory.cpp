#include <memory.h>
#include <vector>
#include <string>
#include <map>

//#define PRINT_MEMORY

typedef struct{
    std::vector<void *> addresses;
    size_t size;
    std::string key;
}Region;

typedef struct{
    std::map<std::string, Region *> globalMap;
    Region *activeEntry;
    std::string activeKey;
    int entries;
}CudaMemoryManager;

const std::string globalKey("global_memory");
static int initialized = 0;
CudaMemoryManager manager;

void CudaMemoryManagerEmpty(){
    manager.activeEntry = nullptr;
    manager.entries = 0;
    manager.activeKey = std::string();
}

void CudaMemoryManagerInsertRegion(const char *key){
    std::string strkey(key);
    Region *entry = new Region;
    entry->size = 0;
    entry->key = key;
    
    manager.globalMap[key] = entry;
    manager.activeEntry = entry;
    manager.activeKey = strkey;
    manager.entries += 1;
#if defined(PRINT_MEMORY)
    std::cout << "[Memory] Active: " << key << std::endl;
#endif
}

void CudaMemoryManagerSelectRegion(const char *key){
    std::string strkey(key);
    if(manager.globalMap.find(strkey) != manager.globalMap.end()){
        manager.activeEntry = manager.globalMap[strkey];
        manager.activeKey = strkey;
#if defined(PRINT_MEMORY)
        std::cout << "[Memory] Active: " << key << std::endl;
#endif
    }
}

void *_cudaAllocator(size_t bytes, int line, const char *filename, bool abort){
    void *ptr = nullptr;
    if(cudaHasMemory(bytes)){
        cudaError_t err = cudaMallocManaged(&ptr, bytes);
        if(err != cudaSuccess){
            std::cout << "Failed to allocate memory " << filename << ":" << line << "[" << bytes << " bytes]" << std::endl;
            ptr = nullptr;
        }else{
            global_memory.allocated += bytes;
        }
    }
    
    if(!ptr && abort){
        getchar();
        cudaSafeExit();
    }
    
    return ptr;
}

__host__ void CudaMemoryManagerClearCurrent(){
    if(manager.activeEntry){
        int size = manager.activeEntry->addresses.size();
#if defined(PRINT_MEMORY)
        size_t bytes = manager.activeEntry->size;
#endif
        for(int i = 0; i < size; i++){
            void *ptr = manager.activeEntry->addresses.at(i);
            cudaFree(ptr);
        }
        
        delete manager.activeEntry;
        manager.globalMap.erase(manager.activeKey);
#if defined(PRINT_MEMORY)
        std::cout << "[Memory] Released: " << manager.activeKey <<
            " [ " << bytes << " ]" << std::endl;
#endif
        manager.activeEntry = nullptr;
        manager.activeKey = std::string();
        CudaMemoryManagerSelectRegion(globalKey.c_str());
    }
}

__host__ void CudaMemoryManagerStart(const char *key){
    std::string strKey(key);
    if(!manager.activeEntry){
        initialized = 1;
        CudaMemoryManagerEmpty();
        CudaMemoryManagerInsertRegion(globalKey.c_str());
    }
    
    if(manager.globalMap.find(strKey) == manager.globalMap.end()){
        CudaMemoryManagerInsertRegion(key);
    }else{
        CudaMemoryManagerSelectRegion(key);
    }
}

void *_cudaAllocate(size_t bytes, int line, const char *filename, bool abort){
    void *ptr = _cudaAllocator(bytes, line, filename, abort);
    if(!initialized){
        initialized = 1;
        CudaMemoryManagerEmpty();
        CudaMemoryManagerInsertRegion(globalKey.c_str());
    }
    manager.activeEntry->addresses.push_back(ptr);
    manager.activeEntry->size += bytes;
    return ptr;
}

__host__ void CudaMemoryManagerClearAll(){
    std::map<std::string, Region *>::iterator it;
    size_t mem = 0;
    for(it = manager.globalMap.begin();
        it != manager.globalMap.end(); )
    {
        Region *region = it->second;
        mem += region->size;
        for(int i = 0; i < region->addresses.size(); i++){
            void *ptr = region->addresses.at(i);
            cudaFree(ptr);
        }
        
        delete region;
        it = manager.globalMap.erase(it);
    }
    
    CudaMemoryManagerEmpty();
    initialized = 0;
#if defined(PRINT_MEMORY)
    std::cout << "[Memory] Released: All [ " << mem <<" ] " << std::endl;
#endif
}
