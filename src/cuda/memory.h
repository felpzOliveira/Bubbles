#pragma once
#include <cutil.h>
/*
* This is a simple memory manager to automatically free all allocs
* performed in the GPU. It creates a new memory context upon calling
* CudaMemoryManagerStart, calls that use the cudaAllocate* family of functions
* will persist the returned pointers in this context, calling 
* CudaMemoryManagerClearCurrent will free all memory taken in the context and
* CudaMemoryManagerClearAll will free all memory taken so far.
* Clearing a memory destroy a context, a global context is provided automatiacally
* to usage but custom contexts need to be recreated.
*/

/*
* Start a new context with the label key, if key already exists switch to
* that context.
*/
void CudaMemoryManagerStart(const char *key);

/*
* Gets the current key being used.
*/
std::string CudaGetCurrentKey();

/*
* Clears all memory taken in the active context.
*/
void CudaMemoryManagerClearCurrent();

/*
* Clears all memory taken so far in all contexts.
*/
void CudaMemoryManagerClearAll();
