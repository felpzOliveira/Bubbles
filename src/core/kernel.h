#pragma once
#include <geometry.h>

class SphStdKernel2{
    public:
    Float h, h2, h3, h4;

    bb_cpu_gpu SphStdKernel2();
    bb_cpu_gpu SphStdKernel2(Float r);
    bb_cpu_gpu void SetRadius(Float r);
    bb_cpu_gpu Float W(Float distance) const;
    bb_cpu_gpu Float dW(Float distance) const;
    bb_cpu_gpu Float d2W(Float distance) const;
    bb_cpu_gpu vec2f gradW(const vec2f &p) const;
    bb_cpu_gpu vec2f gradW(Float distance, const vec2f &v) const;
};

class SphSpikyKernel2{
    public:
    Float h, h2, h3, h4;

    bb_cpu_gpu SphSpikyKernel2();
    bb_cpu_gpu SphSpikyKernel2(Float r);
    bb_cpu_gpu void SetRadius(Float r);
    bb_cpu_gpu Float W(Float distance) const;
    bb_cpu_gpu Float dW(Float distance) const;
    bb_cpu_gpu Float d2W(Float distance) const;
    bb_cpu_gpu vec2f gradW(const vec2f &p) const;
    bb_cpu_gpu vec2f gradW(Float distance, const vec2f &v) const;
};

class SphStdKernel3{
    public:
    Float h, h2, h3, h5;

    bb_cpu_gpu SphStdKernel3();
    bb_cpu_gpu SphStdKernel3(Float r);
    bb_cpu_gpu void SetRadius(Float r);
    bb_cpu_gpu Float W(Float distance) const;
    bb_cpu_gpu Float dW(Float distance) const;
    bb_cpu_gpu Float d2W(Float distance) const;
    bb_cpu_gpu vec3f gradW(const vec3f &p) const;
    bb_cpu_gpu vec3f gradW(Float distance, const vec3f &v) const;
};

class SphSpikyKernel3{
    public:
    Float h, h2, h3, h4, h5;

    bb_cpu_gpu SphSpikyKernel3();
    bb_cpu_gpu SphSpikyKernel3(Float r);
    bb_cpu_gpu void SetRadius(Float r);
    bb_cpu_gpu Float W(Float distance) const;
    bb_cpu_gpu Float dW(Float distance) const;
    bb_cpu_gpu Float d2W(Float distance) const;
    bb_cpu_gpu vec3f gradW(const vec3f &p) const;
    bb_cpu_gpu vec3f gradW(Float distance, const vec3f &v) const;
};

bb_cpu_gpu bool IsWithinSpiky(Float distance, Float radius);
bb_cpu_gpu bool IsWithinStd(Float distance, Float radius);
