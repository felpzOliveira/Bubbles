/* date = November 21st 2020 8:55 pm */

#ifndef PROFILER_H
#define PROFILER_H
#define bb_cpu __host__
#define bb_gpu __device__


#define PROFILER_ALL

#if !defined(PROFILER_ALL)
#define ProfilerSetParticle(v, i)
#else
#define ProfilerSetParticle(v, i) _ProfilerSetParticle(v, i)
#endif

#define ProfilerUpdate(v, i) ProfilerSetParticle(v, i)

void   ProfilerInitKernel(int pCount);
void   ProfilerPrepare(const char *funcname);
void   ProfilerBeginStep();
void   ProfilerIncreaseStepIteration();
int    ProfilerGetIterationsPassed();
double ProfilerGetStepInterval();
double ProfilerGetEvaluation(const char *fname);
void   ProfilerEndStep();
void   ProfilerFinish();
void   ProfilerReport(int frameId=-1);
void   ProfilerManualStart(const char *funcname);
void   ProfilerManualFinish();

bb_cpu bb_gpu
void _ProfilerSetParticle(int value, int id);


#endif //PROFILER_H
