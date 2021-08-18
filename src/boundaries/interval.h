/* date = May 19th 2021 15:25 */
#pragma once
#include <grid.h>
#include <cutil.h>

/**************************************************************/
//               I N T E R V A L   M E T H O D                //
//                      Trivial version                       //
/**************************************************************/

/*
* Attempting to implement Sandim's new interval method for boundary detection.
* Going to implement the simple one first and we see where this goes.
*/

//TODO: This is not GPU ready. Heavily recursive.
template<typename T>
inline __bidevice__ void IntervalBoundaryGetBoundsFor2(ParticleSet<T> *pSet, int i,
                                                       Bounds2f *bounds, Float h)
{
    vec2f p = pSet->GetParticlePosition(i);
    *bounds = Bounds2f(p - vec2f(h), p + vec2f(h));
}

template<typename T>
__bidevice__ bool IntervalQuery2(ParticleSet<T> *pSet, int i, Bounds2f Bq,
                                 int depth, int max_depth, Float h)
{
    bool interior = true;
    Bucket *bucket = pSet->GetParticleBucket(i);
    for(int k = 0; k < bucket->Count(); k++){
        Bounds2f Bj;
        int j = bucket->Get(k);

        if(i == j) continue;

        IntervalBoundaryGetBoundsFor2(pSet, j, &Bj, h);
        // in this case this Q can be let go
        if(Inside(Bq, Bj)){
            return true;
        }

        // if Q is not inside Bj, overlaps will only return the 'straddle' class.
        if(Overlaps(Bq, Bj)){
            interior = false;
        }
    }

    if(interior){
        return false;
    }

    if(depth >= max_depth){
        return false;
    }

    Bounds2f inner[4];
    int s = SplitBounds(Bq, &inner[0]);
    bool r = true;
    for(int k = 0; k < s; k++){
        r &= IntervalQuery2(pSet, i, inner[k], depth+1, max_depth, h);
    }

    return r;
}

template<typename T>
__bidevice__ void IntervalBoundaryClassify2(ParticleSet<T> *pSet, int i, Float h){
    Bounds2f Bi;
    int max_depth = 1;
    IntervalBoundaryGetBoundsFor2(pSet, i, &Bi, h);
    if(!IntervalQuery2(pSet, i, Bi, 0, max_depth, h)){
        pSet->SetParticleV0(i, 1);
    }
}

template<typename T>
__host__ void IntervalBoundary2(ParticleSet<T> *pSet, Float h){
    int N = pSet->GetParticleCount();

    // this is a per-particle method
    for(int i = 0; i < N; i++){
        IntervalBoundaryClassify2(pSet, i, h);
    }
}
