#include <kernel.h>

bb_cpu_gpu SphStdKernel2::SphStdKernel2():h(0), h2(0), h3(0), h4(0){}
bb_cpu_gpu SphStdKernel2::SphStdKernel2(Float r){ SetRadius(r); }

bb_cpu_gpu void SphStdKernel2::SetRadius(Float r){
    h = r;
    h2 = h * h;
    h3 = h2 * h;
    h4 = h2 * h2;
}

bb_cpu_gpu Float SphStdKernel2::W(Float distance) const{
    Float d2 = distance * distance;
    Float of = d2 - h2;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - d2 / h2;
        return 4.0 / (Pi * h2) * x * x * x;
    }
}

bb_cpu_gpu Float SphStdKernel2::dW(Float distance) const{
    Float d2 = distance * distance;
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - d2 / h2;
        return -24.0 * distance / (Pi * h4) * x * x;
    }
}

bb_cpu_gpu Float SphStdKernel2::d2W(Float distance) const{
    Float d2 = distance * distance;
    Float of = d2 - h2;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = d2 / h2;
        return 24.0 / (Pi * h4) * (1 - x) * (5 * x - 1);
    }
}

bb_cpu_gpu vec2f SphStdKernel2::gradW(Float distance, const vec2f &v) const{
    return -dW(distance) * v;
}

bb_cpu_gpu vec2f SphStdKernel2::gradW(const vec2f &p) const{
    Float dist = p.Length();
    if(dist > 0 && !IsZero(dist)){
        return gradW(dist, p / dist);
    }else{
        return vec2f(0,0);
    }
}

bb_cpu_gpu SphSpikyKernel2::SphSpikyKernel2():h(0), h2(0), h3(0), h4(0){}
bb_cpu_gpu SphSpikyKernel2::SphSpikyKernel2(Float r){ SetRadius(r); }

bb_cpu_gpu void SphSpikyKernel2::SetRadius(Float r){
    h = r;
    h2 = h * h;
    h3 = h2 * h;
    h4 = h2 * h2;
}

bb_cpu_gpu Float SphSpikyKernel2::W(Float distance) const{
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - distance / h;
        return 10.0 / (Pi * h2) * x * x * x;
    }
}

bb_cpu_gpu Float SphSpikyKernel2::dW(Float distance) const{
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - distance / h;
        return -30.0 / (Pi * h3) * x * x;
    }
}

bb_cpu_gpu Float SphSpikyKernel2::d2W(Float distance) const{
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - distance / h;
        return 60.0 / (Pi * h4) * x;
    }
}

bb_cpu_gpu vec2f SphSpikyKernel2::gradW(Float distance, const vec2f &v) const{
    return -dW(distance) * v;
}

bb_cpu_gpu vec2f SphSpikyKernel2::gradW(const vec2f &p) const{
    Float dist = p.Length();
    if(dist > 0 && !IsZero(dist)){
        return gradW(dist, p / dist);
    }else{
        return vec2f(0,0);
    }
}

bb_cpu_gpu SphStdKernel3::SphStdKernel3():h(0), h2(0), h3(0), h5(0){}
bb_cpu_gpu SphStdKernel3::SphStdKernel3(Float r){ SetRadius(r); }

bb_cpu_gpu void SphStdKernel3::SetRadius(Float r){
    AssertA(!IsZero(r), "Invalid kernel spacing");
    h = r;
    h2 = h * h;
    h3 = h2 * h;
    h5 = h2 * h3;
}

bb_cpu_gpu Float SphStdKernel3::W(Float distance) const{
    Float d2 = distance * distance;
    Float of = d2 - h2;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - d2 / h2;
        return 315.0 / (64.0 * Pi * h3) * x * x * x;
    }
}

bb_cpu_gpu Float SphStdKernel3::dW(Float distance) const{
    Float d2 = distance * distance;
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - d2 / h2;
        return -945.0 / (32.0 * Pi * h5) * distance * x * x;
    }
}

bb_cpu_gpu Float SphStdKernel3::d2W(Float distance) const{
    Float d2 = distance * distance;
    Float of = d2 - h2;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = d2 / h2;
        return 945.0 / (32.0 * Pi * h5) * (1 - x) * (5 * x - 1);
    }
}

bb_cpu_gpu vec3f SphStdKernel3::gradW(const vec3f &p) const{
    Float dist = p.Length();
    if(dist > 0 && !IsZero(dist)){
        return gradW(dist, p / dist);
    }else{
        return vec3f(0,0,0);
    }
}

bb_cpu_gpu  vec3f SphStdKernel3::gradW(Float distance, const vec3f &v) const{
    return -dW(distance) * v;
}

bb_cpu_gpu SphSpikyKernel3::SphSpikyKernel3():h(0), h2(0), h3(0), h5(0){}
bb_cpu_gpu SphSpikyKernel3::SphSpikyKernel3(Float r){ SetRadius(r); }

bb_cpu_gpu void SphSpikyKernel3::SetRadius(Float r){
    AssertA(!IsZero(r), "Invalid kernel spacing");
    h = r;
    h2 = h * h;
    h3 = h2 * h;
    h4 = h2 * h2;
    h5 = h2 * h3;
}

bb_cpu_gpu Float SphSpikyKernel3::W(Float distance) const{
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - distance / h;
        return 15.0 / (Pi * h3) * x * x * x;
    }
}

bb_cpu_gpu Float SphSpikyKernel3::dW(Float distance) const{
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - distance / h;
        return -45.0 / (Pi * h4) * x * x;
    }
}

bb_cpu_gpu Float SphSpikyKernel3::d2W(Float distance) const{
    Float of = distance - h;
    if(IsZero(of) || of > 0){
        return 0;
    }else{
        Float x = 1.0 - distance / h;
        return 90.0 / (Pi * h5) * x;
    }
}

bb_cpu_gpu vec3f SphSpikyKernel3::gradW(const vec3f &p) const{
    Float dist = p.Length();
    if(dist > 0 && !IsZero(dist)){
        return gradW(dist, p / dist);
    }else{
        return vec3f(0,0,0);
    }
}

bb_cpu_gpu  vec3f SphSpikyKernel3::gradW(Float distance, const vec3f &v) const{
    return -dW(distance) * v;
}

bb_cpu_gpu bool IsWithinSpiky(Float distance, Float radius){
    Float of = distance - radius;
    return !(IsZero(of) || of > 0);
}

bb_cpu_gpu bool IsWithinStd(Float distance, Float radius){
    Float h2 = radius * radius;
    Float d2 = distance * distance;
    Float of = d2 - h2;
    return !(IsZero(of) || of > 0);
}
