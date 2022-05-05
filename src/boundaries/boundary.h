/* date = August 18th 2021 21:29 */
#pragma once
#include <string>
#include <vector>
#include <colorfield.h>
#include <doring.h>
#include <lnm.h>
#include <dilts.h>
#include <xiaowei.h>
#include <interval.h>
#include <sandim.h>
#include <marrone.h>

typedef enum{
    BOUNDARY_LNM=0,
    BOUNDARY_DILTS,
    BOUNDARY_COLOR_FIELD,
    BOUNDARY_DORING,
    BOUNDARY_XIAOWEI,
    BOUNDARY_SANDIM,
    BOUNDARY_INTERVAL,
    BOUNDARY_MARRONE,
    BOUNDARY_NONE
}BoundaryMethod;

inline std::string GetBoundaryMethodName(BoundaryMethod method){
    switch(method){
        case BOUNDARY_LNM : return "Lnm";
        case BOUNDARY_DILTS : return "Dilts";
        case BOUNDARY_COLOR_FIELD : return "ColorField";
        case BOUNDARY_DORING: return "Doring";
        case BOUNDARY_XIAOWEI : return "Xiaowei";
        case BOUNDARY_SANDIM : return "Sandim";
        case BOUNDARY_INTERVAL : return "Interval";
        case BOUNDARY_MARRONE: return "Marrone";
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
    if(method == "ColorField" || method == "Colorfield" || method == "colorfield" ||
       method == "CF" || method == "Cf" || method == "cf")
        return BOUNDARY_COLOR_FIELD;
    if(method == "XIAOWEI" || method == "Xiaowei" || method == "xiaowei")
        return BOUNDARY_XIAOWEI;
    if(method == "SANDIM" || method == "Sandim" || method == "sandim")
        return BOUNDARY_SANDIM;
    if(method == "LNM" || method == "Lnm" || method == "lnm")
        return BOUNDARY_LNM;
    if(method == "Doring" || method == "doring" || method == "RDM")
        return BOUNDARY_DORING;
    if(method == "Marrone" || method == "marrone" || method == "MARRONE")
        return BOUNDARY_MARRONE;

    return BOUNDARY_NONE;
}

inline void GetBoundaryNames(std::vector<std::string> &names){
    int s0 = 0;
    int sn = (int)BOUNDARY_NONE;
    for(int i = s0; i < sn; i++){
        BoundaryMethod met = (BoundaryMethod)i;
        names.push_back(GetBoundaryMethodName(met));
    }
}

