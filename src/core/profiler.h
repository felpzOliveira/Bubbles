/* date = November 21st 2020 8:55 pm */

#ifndef PROFILER_H
#define PROFILER_H

#define PROFILER_ALL

#if !defined(PROFILER_ALL)
#define ProfilerSetParticle(v, i)
#else
#define ProfilerSetParticle(v, i) _ProfilerSetParticle(v, i)
#endif

#define ProfilerUpdate(v, i) ProfilerSetParticle(v, i)

void ProfilerInitKernel(int pCount);
void ProfilerPrepare(const char *funcname);
void ProfilerFinish();
void ProfilerReport();

__host__ __device__ 
void _ProfilerSetParticle(int value, int id);


#endif //PROFILER_H
