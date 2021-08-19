/* date = August 18th 2021 21:29 */
#pragma once
#include <string>
#include <vector>
#include <muller.h>
#include <lnm.h>
#include <dilts.h>
#include <xiaowei.h>
#include <interval.h>

typedef enum{
    BOUNDARY_LNM=0,
    BOUNDARY_DILTS,
    BOUNDARY_MULLER,
    BOUNDARY_XIAOWEI,
    BOUNDARY_INTERVAL,
    BOUNDARY_NONE
}BoundaryMethod;

inline std::string GetBoundaryMethodName(BoundaryMethod method){
    switch(method){
        case BOUNDARY_LNM : return "Lnm";
        case BOUNDARY_DILTS : return "Dilts";
        case BOUNDARY_MULLER : return "Muller";
        case BOUNDARY_XIAOWEI : return "Xiaowei";
        case BOUNDARY_INTERVAL : return "Interval";
        default:{
            return "NONE";
        }
    }
}

inline BoundaryMethod GetBoundaryMethod(std::string method){
    if(method == "DILTS" || method == "Dilts" || method == "dilts")
        return BOUNDARY_DILTS;
    if(method == "INTERVAL" || method == "Interval" || method == "interval")
        return BOUNDARY_INTERVAL;
    if(method == "MULLER" || method == "Muller" || method == "muller")
        return BOUNDARY_MULLER;
    if(method == "XIAOWEI" || method == "Xiaowei" || method == "xiaowei")
        return BOUNDARY_XIAOWEI;

    return BOUNDARY_LNM;
}

inline void GetBoundaryNames(std::vector<std::string> &names){
    int s0 = 0;
    int sn = (int)BOUNDARY_NONE;
    for(int i = s0; i < sn; i++){
        BoundaryMethod met = (BoundaryMethod)i;
        names.push_back(GetBoundaryMethodName(met));
    }
}

