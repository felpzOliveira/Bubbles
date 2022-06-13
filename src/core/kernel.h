#pragma once
#include <geometry.h>

class SphStdKernel2{
    public:
    Float h, h2, h3, h4;

    __bidevice__ SphStdKernel2();
    __bidevice__ SphStdKernel2(Float r);
    __bidevice__ void SetRadius(Float r);
    __bidevice__ Float W(Float distance) const;
    __bidevice__ Float dW(Float distance) const;
    __bidevice__ Float d2W(Float distance) const;
    __bidevice__ vec2f gradW(const vec2f &p) const;
    __bidevice__ vec2f gradW(Float distance, const vec2f &v) const;
};

class SphSpikyKernel2{
    public:
    Float h, h2, h3, h4;

    __bidevice__ SphSpikyKernel2();
    __bidevice__ SphSpikyKernel2(Float r);
    __bidevice__ void SetRadius(Float r);
    __bidevice__ Float W(Float distance) const;
    __bidevice__ Float dW(Float distance) const;
    __bidevice__ Float d2W(Float distance) const;
    __bidevice__ vec2f gradW(const vec2f &p) const;
    __bidevice__ vec2f gradW(Float distance, const vec2f &v) const;
};

class SphStdKernel3{
    public:
    Float h, h2, h3, h5;

    __bidevice__ SphStdKernel3();
    __bidevice__ SphStdKernel3(Float r);
    __bidevice__ void SetRadius(Float r);
    __bidevice__ Float W(Float distance) const;
    __bidevice__ Float dW(Float distance) const;
    __bidevice__ Float d2W(Float distance) const;
    __bidevice__ vec3f gradW(const vec3f &p) const;
    __bidevice__ vec3f gradW(Float distance, const vec3f &v) const;
};

class SphSpikyKernel3{
    public:
    Float h, h2, h3, h4, h5;

    __bidevice__ SphSpikyKernel3();
    __bidevice__ SphSpikyKernel3(Float r);
    __bidevice__ void SetRadius(Float r);
    __bidevice__ Float W(Float distance) const;
    __bidevice__ Float dW(Float distance) const;
    __bidevice__ Float d2W(Float distance) const;
    __bidevice__ vec3f gradW(const vec3f &p) const;
    __bidevice__ vec3f gradW(Float distance, const vec3f &v) const;
};

__bidevice__ bool IsWithinSpiky(Float distance, Float radius);
__bidevice__ bool IsWithinStd(Float distance, Float radius);
