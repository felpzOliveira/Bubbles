/* date = May 5th 2021 13:4 */
#pragma once
#include <shape.h>
#include <geometry.h>
#include <cutil.h>

/*
* Defines utility sdf functions for generating sdf shapes. These can be used
* for generating coliders/emitters with complex shapes. If you wish to combine
* sdfs and make more complex shapes you can add your own sdf computation here.
*/

#define SDF_Sphere(center, radius) GPU_LAMBDA(vec3f point, Shape *, int) -> Float{\
    return Distance(point, center) - radius;\
}

#define SDF_Torus(center, radius) GPU_LAMBDA(vec3f point, Shape *, int) -> Float{\
    vec3f p = point - center;\
    vec2f xz(p.x, p.z);\
    vec2f q(xz.Length() - radius.x, p.y);\
    return q.Length() - radius.y;\
}

#define SDF_RoundBox(center, length, radius) GPU_LAMBDA(vec3f point, Shape *, int) -> Float{\
    vec3f p = point - center;\
    vec3f q = Abs(p) - length;\
    vec3f a(Max(q.x, 0), Max(q.y, 0), Max(q.z, 0));\
    Float f = Max(q.x, Max(q.y, q.z));\
    return a.Length() + Min(f, 0.0) - radius;\
}

/*
* The following is the implementation of the teddy scenes.
* The only reason this is here is because I used Bubbles for
* the marching cubes implementation and since we don't have any
* external compiler for this I'll just implement the models here.
* Setup is:
* Make a bounding box from -5 to 5 with spacing 0.02, use CreateSDF routine from shapes.h
* and sample the Teddy functions. Once the grid is built simply pass the results
* to marching cubes.
*/

inline bb_cpu_gpu vec3f Zrot(vec3f v, Float angle){
    Float ang = Radians(angle);
    Float si = std::sin(ang), co = std::cos(ang);
    return vec3f(
        v.x * co - v.y * si,
        v.x * si + v.y * co,
        v.z
    );
}

inline bb_cpu_gpu vec3f Yrot(vec3f v, Float angle){
    Float ang = Radians(angle);
    Float si = std::sin(ang), co = std::cos(ang);
    return vec3f(
        v.x * co + v.z * si,
        v.y,
        -si * v.x + v.z * co
    );
}

inline bb_cpu_gpu vec3f Xrot(vec3f v, Float angle){
    Float ang = Radians(angle);
    Float si = std::sin(ang), co = std::cos(ang);
    return vec3f(
        v.x,
        v.y * co - v.z * si,
        v.y * si + v.z * co
    );
}

inline bb_cpu_gpu Float SMax(Float a, Float b, Float k){
    Float h = Clamp( 0.5 + 0.5*(b - a)/k, 0.0, 1.0);
    return Mix(a, b, h) + k * h * (1.0 - h);
}

inline bb_cpu_gpu Float SMin(Float a, Float b, Float k){
    Float h = Clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0);
    return Mix(b, a, h) - k*h*(1.0-h);
}

// s => vec4(center.xyz, radius)
inline bb_cpu_gpu Float T_Sphere(vec3f p, vec4f s){
    return (p - vec3f(s.x, s.y, s.z)).Length() - s.w;
}

// b => vec3(sizeX, sizeY, sizeZ)
inline bb_cpu_gpu Float T_Box(vec3f p, vec3f b, Float r){
    vec3f q = Abs(p) - b;
    q = vec3f(Max(q.x, 0), Max(q.y, 0), Max(q.z, 0));
    return q.Length() - r;
}

// a => min point, b => max point
inline bb_cpu_gpu Float T_Capsule(vec3f p, vec3f a, vec3f b, Float r){
    vec3f pa = p - a;
    vec3f ba = b - a;
    Float h = Clamp(Dot(pa, ba) / Dot(ba, ba), 0.0, 1.0);
    return (pa - ba * h).Length() - r;
}

// c => center, r => radius on 3 axis
inline bb_cpu_gpu Float T_Elipse(vec3f p, vec3f c, vec3f r){
    vec3f pc = (p - c);
    pc.x /= r.x;
    pc.y /= r.y;
    pc.z /= r.z;
    return (pc.Length() - 1.0f) * Min(Min(r.x, r.y), r.z);
}

inline bb_cpu_gpu Float Teddy_Lying(vec3f point){
    vec3f p = point;
    //p.y += 0.6;
    //p.z += 3.0;

    vec3 q = p;
    q = Xrot(q, 80.0);
    q = Zrot(q, -80.0);
    q.x = abs(q.x);
    Float d0 = T_Sphere(q, vec4f(0.0, 1.4, 0.0, 1.2));
    Float d1 = T_Box(q - vec3f(0.0, 1.4, 0.0), vec3f(1.0, 0.45, 0.45), 0.9);
    vec3 sq = q;

    q = sq;
    Float eye  = T_Elipse(q, vec3f(0.4, 1.4, 0.92), vec3f(0.1, 0.31, 0.1));
    Float head = SMax(d0, d1, 1.1);

    Float d2 = T_Capsule(q, vec3f(0.0, 2.2, 0.0), vec3f(0.0, 2.6,0.0), 0.10);
    Float d3 = T_Sphere(q, vec4f(0.0, 2.69, 0.0, 0.21));
    Float antena = SMin(d2,d3, 0.1);

    q = Zrot(q, 180.0);
    q.y -= 0.9;
    q.z += 0.6;
    Float d4 = T_Elipse(q, vec3f(0.0, 0.0, 0.0), vec3f(0.80, 1.50, 0.80));
    Float d5 = T_Sphere(q, vec4f(0.0, 1.4, 0.0, 1.2));
    Float body = SMax(d4, -d5, 1.0);

    q = p; q.z = Absf(q.z);
    q.y += 0.5; q.x += 0.3;

    Float d7 = T_Sphere(q, vec4f(0.4, -0.3, 1.0, 0.3));
    Float d10 = T_Capsule(q, vec3f(0.1, 0.1, 0.0), vec3f(0.4, -0.3, 0.95), 0.18);
    Float arm = SMin(d10, d7, 0.1);

    q = p; q.z = Absf(q.z);
    q.y -= 0.75; q.x -= 0.93;
    Float d8 = T_Capsule(q, vec3f(0.1, -1.0, 0.4), vec3f(0.3, -1.2, 0.4), 0.18);

    Float d9 = T_Sphere(q, vec4f(0.35, -1.5, 0.4, 0.3));

    Float foot = SMin(d8, d9, 0.1);

    Float d = SMin(SMin(antena, body, 0.1), head, 0.1);
    d = SMin(SMin(d, arm, 0.1), foot, 0.1);

    d = Min(d, eye);
    return d;
}

inline bb_cpu_gpu Float Teddy_Standing(vec3f point){
    vec3f p = point;
    vec3f s = p;

    p = Zrot(p, -30.0);
    p = Yrot(p, 50.0);

    vec3f q = p;
    q.x = Absf(q.x);
    Float d0 = T_Sphere(q, vec4f(0.0, 1.4, 0.0, 1.2));
    Float d1 = T_Box(q - vec3f(0.0, 1.4, 0.0), vec3f(1.0, 0.45, 0.45), 0.9);
    Float eye = T_Elipse(q, vec3f(0.4, 1.4, 0.92), vec3f(0.1,0.16 + 0.15 ,0.1));

    Float head = SMax(d0, d1, 1.1);

    Float d2 = T_Capsule(q, vec3f(0.0, 2.2, 0.0), vec3f(0.0, 2.6,0.0), 0.10);
    Float d3 = T_Sphere(q, vec4f(0.0, 2.69, 0.0, 0.21));
    Float antena = SMin(d2, d3, 0.1);

    s = Yrot(s, 50.0);
    q = s; q.x = Absf(q.x);

    q = Zrot(q, 180.0);
    q.y -= 0.9;
    Float d4 = T_Elipse(q, vec3f(0.0, 0.0, 0.0), vec3f(0.80, 1.50, 0.80));
    Float d5 = T_Sphere(q, vec4f(0.0, 1.4, 0.0, 1.2));
    Float body = SMax(d4, -d5, 1.0);

    q = s; q.x = Absf(q.x);
    vec3 k = q;
    Float d6 = T_Capsule(q, vec3f(0.4, 0.1, 0.0), vec3f(1.0, -0.2, 0.5), 0.18);
    Float d7 = T_Sphere(k, vec4f(1.0, -0.2, 0.5, 0.3));
    Float arm = SMin(d6, d7, 0.0);
    arm = SMin(d6, d7, 0.1);

    Float d8 = T_Capsule(q, vec3f(0.3, -1.1, 0.0), vec3f(0.3, -1.2, 0.0), 0.18);
    Float d9 = T_Sphere(q, vec4f(0.35, -1.5, 0.0, 0.3));
    Float d10 = T_Box(q + vec3f(-0.35, 1.5, 0.0), vec3f(0.1, 0.1, 0.1), 0.15);
    Float d11 = SMin(d9, d10, 0.1);

    Float foot = SMin(SMin(d8, d9, 0.1), d11, 0.1);
    Float d = SMin(SMin(antena, body, 0.1), head, 0.1);
    d = SMin(SMin(d, arm, 0.1), foot, 0.1);
    //d = Min(d, eye);
    return d;
}

inline bb_cpu_gpu Float Teddy_Sitting(vec3f point){
    vec3f p = point;
    //p.x -= 4.5;
    //p.y += 0.6;
    //p.z += 3.0;

    vec3f q = p;

    q.x = Absf(q.x);
    Float d0 = T_Sphere(q, vec4f(0.0, 1.4, 0.0, 1.2));
    Float d1 = T_Box(q - vec3f(0.0, 1.4, 0.0), vec3f(1.0, 0.45, 0.45), 0.9);
    Float eye = T_Elipse(q, vec3f(0.4, 1.4, 0.92), vec3f(0.1,0.16 + 0.15 ,0.1));
    Float head = SMax(d0, d1, 1.1);

    Float d2 = T_Capsule(q, vec3f(0.0, 2.2, 0.0), vec3f(0.0, 2.6,0.0), 0.10);
    Float d3 = T_Sphere(q, vec4f(0.0, 2.69, 0.0, 0.21));
    Float antena = SMin(d2, d3, 0.1);

    q = p; q.x = Absf(q.x);
    q = Zrot(q, 180.0);
    q.y -= 0.9;
    Float d4 = T_Elipse(q, vec3f(0.0, 0.0, 0.0), vec3f(0.80, 1.50, 0.80));
    Float d5 = T_Sphere(q, vec4f(0.0, 1.4, 0.0, 1.2));
    Float body = SMax(d4, -d5, 1.0);

    q = p; q.x = Absf(q.x);
    Float d6 = T_Capsule(q, vec3f(0.4, 0.1, 0.0), vec3f(0.7, -0.2, 0.5), 0.18);
    Float d12 = T_Capsule(q, vec3f(0.7,-0.2,0.5), vec3f(0.2, -0.2, 0.9), 0.18);
    Float d7 = T_Sphere(q, vec4f(0.2, -0.2, 0.9, 0.3));
    d6 = SMin(d6, d12, 0.1);
    Float arm = SMin(d6, d7, 0.0);
    arm = SMin(d6, d7, 0.1);

    q = Xrot(q, 40.0);
    Float d8 = T_Capsule(q, vec3f(0.3, -1.0, 0.1), vec3f(0.3, -1.2, 0.6), 0.18);
    Float d9 = T_Sphere(q, vec4f(0.35, -1.5, 0.6, 0.3));
    Float foot = SMin(d8, d9, 0.1);

    Float d = SMin(SMin(antena, body, 0.1), head, 0.1);
    d = SMin(SMin(d, arm, 0.1), foot, 0.1);
    d = Min(d, eye);

    return d;
}

/*
* The following is the implementation of the origami scenes.
* Like before it is only here so we can save this code for future reference if we ever
* need. Setup is:
* Make a bounding box from -5 to 5 with spacing 0.02,use CreateSDF
* routine from shapes.h and sample the each functions. Once the grid is built simply
* pass the results to marching cubes using an isovalue = 0.04. Some objects have an
* id variable into their SDF, this is only so we can only output a part of the object
* for giving specific materials for parts of the resulting mesh. This one does not
* use smooth unions.
*/
inline bb_cpu_gpu Float T_Dot2(const vec3f &v){
    return Dot(v, v);
}

inline bb_cpu_gpu Float T_Triangle(vec3f p, vec3f a, vec3f b, vec3f c){
    vec3f ba = b - a; vec3f pa = p - a;
    vec3f cb = c - b; vec3f pb = p - b;
    vec3f ac = a - c; vec3f pc = p - c;
    vec3f nor = Cross(ba, ac);

    Float sa = Sign(Dot(Cross(ba, nor), pa));
    Float sb = Sign(Dot(Cross(cb, nor), pb));
    Float sc = Sign(Dot(Cross(ac, nor), pc));
    Float ss = sa + sb + sc;
    if(ss < 2){
        Float a2 = T_Dot2(ba * Clamp(Dot(ba, pa) / T_Dot2(ba), 0.0, 1.0) - pa);
        Float b2 = T_Dot2(cb * Clamp(Dot(cb, pb) / T_Dot2(cb), 0.0, 1.0) - pb);
        Float c2 = T_Dot2(ac * Clamp(Dot(ac, pc) / T_Dot2(ac), 0.0, 1.0) - pc);
        return sqrt(Min(a2, Min(b2, c2)));
    }else{
        return sqrt(Dot(nor,pa) * Dot(nor, pa) / T_Dot2(nor));
    }
}

inline bb_cpu_gpu Float T_OrigamiBoat(vec3f point, int id){
    vec3f p = point;
    vec3f half(5.0);
    vec3f q = vec3f(Absf(p.x), p.y, Absf(p.z));

    vec3f Poo(0.f, -half.y * 0.9, 0.f);
    vec3f P0(half.x*0.1, -half.y * 0.9, half.z * 0.6);
    vec3f P1(half.x*0.1, -half.y * 0.9, half.z * 0.01 + 0.03f);
    vec3f P2(0.01f, half.y * 0.95, 0.01f  + 0.03f);
    vec3f Pk(half.x * 0.95, 0.f, half.z * 0.01 + 0.03f);
    vec3f Pn(0.01f, half.y * 0.1, half.z * 0.9);

    Float tri1 = T_Triangle(q, P0, P1, P2);
    Float tri2 = T_Triangle(q, P0, Pk, Poo);
    Float tri3 = T_Triangle(q, P0, Pk, Pn);
    vec2f tris[3] = { vec2f(tri1, 0), vec2f(tri2, 1), vec2f(tri3, 2) };
    vec2f sdf(Infinity, -1);
    for(int i = 0; i < 3; i++){
        bool accept = id < 0 || id == tris[i].y;
        if(sdf.x > tris[i].x && accept){
            sdf = tris[i];
        }
    }

    return sdf.x;
}

inline bb_cpu_gpu Float T_OrigamiDragon(vec3f point){
    vec3f p = point;
    vec3f half(5.0);
    vec3f q(p.x, p.y, Absf(p.z));
    vec3f P0   (-half.x * 0.70,  half.y * 0.85 ,half.z * 0.05);
    vec3f P1   (-half.x * 0.25,  half.y * 0.40, half.z * 0.06);
    vec3f P2   (-half.x * 0.55, -half.y * 0.15, half.z * 0.05);
    vec3f P3   (+half.x * 0.10, -half.y * 0.60, half.z * 0.09);
    vec3f P4   (+half.x * 0.20,  half.y * 0.15, half.z * 0.08);
    vec3f P5Zp (+half.x * 0.05,  half.y * 0.25, half.z * 0.00);
    vec3f P6Zp (+half.x * 0.35,  half.y * 0.10, half.z * 0.02);
    vec3f P7Zp (+half.x * 0.45, -half.y * 0.60, half.z * 0.05);
    vec3f P8Zp (-half.x * 0.35, -half.y * 0.60, half.z * 0.05);
    vec3f P2Sh (-half.x * 0.55, -half.y * 0.15, half.z * 0.05);
    vec3f P9pp (+half.x * 0.40,  half.y * 0.20, half.z * 0.00);
    vec3f P10pp(+half.x * 0.70, -half.y * 0.75, half.z * 0.10);
    vec3f P11pp(+half.x * 0.43, -half.y * 0.96, half.z * 0.15);
    vec3f P12m (-half.x * 0.65, -half.y * 0.96, half.z * 0.15);
    vec3f P13m (-half.x * 0.45, -half.y * 0.05, half.z * 0.05);
    vec3f P14p (+half.x * 0.80,  half.y * 0.10, half.z * 0.01);
    vec3f P15p (+half.x * 0.50, -half.y * 0.60, half.z * 0.01);
    vec3f Q3   (+half.x * 0.60,  half.y * 0.10, half.z * 0.01);
    vec3f Q2   (+half.x * 0.55,  half.y * 0.85, half.z * 0.03);
    vec3f Q0   (+half.x * 0.70,  half.y * 0.90, half.z * 0.03);
    vec3f Q6   (+half.x * 0.71,  half.y * 0.70, half.z * 0.10);
    vec3f Q4   (+half.x * 0.90,  half.y * 0.90, half.z * 0.00);
    vec3f Q5   (+half.x * 0.88,  half.y * 0.80, half.z * 0.00);
    vec3f Q7   (+half.x * 0.65,  half.y * 1.00, half.z * 0.00);
    vec3f Q8   (+half.x * 0.45,  half.y * 1.00, half.z * 0.00);
    vec3f Q9   (+half.x * 0.64,  half.y * 0.88, half.z * 0.03);
    vec3f T0   (-half.x * 0.85, -half.y * 0.25, half.z * 0.03);
    vec3f T1   (-half.x * 0.80, -half.y * 0.50, half.z * 0.03);
    vec3f T2   (-half.x * 0.83,  half.y * 0.16, half.z * 0.01);
    vec3f T3   (-half.x * 1.00,  half.y * 0.19, half.z * 0.01);
    vec3f T4   (-half.x * 0.53, -half.y * 0.23, half.z * 0.01);
    vec3f T5   (-half.x * 0.90,  half.y * 0.75, half.z * 0.00);

    Float tri1  = T_Triangle(q, P0, P1, P2);
    Float tri2  = T_Triangle(q, P1, P2, P3);
    Float tri3  = T_Triangle(q, P1, P3, P4);
    Float tri4  = T_Triangle(q, P5Zp, P6Zp, P7Zp);
    Float tri5  = T_Triangle(q, P5Zp, P7Zp, P8Zp);
    Float tri6  = T_Triangle(q, P5Zp, P8Zp, P2Sh);
    Float tri7  = T_Triangle(q, P9pp, P6Zp, P10pp);
    Float tri8  = T_Triangle(q, P6Zp, P11pp, P10pp);
    Float tri9  = T_Triangle(q, P8Zp, P12m, P13m);
    Float tri10 = T_Triangle(q, P15p, P14p, P6Zp);
    Float tri11 = T_Triangle(q, Q3, Q2, P14p);
    Float tri12 = T_Triangle(q, Q0, P14p, Q2);
    Float tri13 = T_Triangle(q, Q6, Q5, Q0);
    Float tri14 = T_Triangle(q, Q5, Q4, Q0);
    Float tri15 = T_Triangle(q, Q4, Q7, Q9);
    Float tri16 = T_Triangle(q, Q7, Q9, Q8);
    Float tri17 = T_Triangle(q, P9pp, T0, T1);
    Float tri18 = T_Triangle(q, T1, P9pp, P7Zp);
    Float tri19 = T_Triangle(q, T0, T2, T3);
    Float tri20 = T_Triangle(q, T0, T2, T4);
    Float tri21 = T_Triangle(q, T3, T2, T5);

    Float sdf = Min(Min(tri1, tri2), tri3);
    sdf = Min(Min(sdf, tri4), tri5);
    sdf = Min(Min(sdf, tri6), tri7);
    sdf = Min(Min(sdf, tri8), tri9);
    sdf = Min(Min(sdf, tri10), tri11);
    sdf = Min(Min(sdf, tri12), tri13);
    sdf = Min(Min(sdf, tri14), tri15);
    sdf = Min(Min(sdf, tri16), tri17);
    sdf = Min(Min(sdf, tri18), tri19);
    sdf = Min(Min(sdf, tri20), tri21);
    return sdf;
}

inline bb_cpu_gpu Float T_OrigamiWhale(vec3f point, int id){
    vec3f half(5.0, 2.0, 0.5);
    vec3f p = point;
    vec3f q(p.x, p.y, Absf(p.z));

    vec3f Q0 (-half.x * 0.15,  half.y * 0.99, half.z * 0.20);
    vec3f Q1 (-half.x * 0.65, -half.y * 0.05, half.z * 0.80);
    vec3f Q2 (-half.x * 0.10, -half.y * 0.80, half.z * 0.99);

    vec3f Q3 (-half.x * 0.98,  half.y * 0.82, half.z * 0.40);
    vec3f Q4 (+half.x * 0.98, -half.y * 0.15, half.z * 0.03);
    vec3f Q5 (+half.x * 0.90, -half.y * 0.80, half.z * 0.70);
    vec3f Q6 (-half.x * 0.98,  half.y * 0.15, half.z * 0.50);
    vec3f Q7 (+half.x * 0.85, -half.y * 0.99, half.z * 0.80);
    vec3f Q8 (-half.x * 0.80, -half.y * 0.99, half.z * 0.80);
    vec3f Q9 (-half.x * 0.96, -half.y * 0.76, half.z * 0.70);
    vec3f Q10(+half.x * 0.95,  half.y * 0.85, half.z * 0.00);
    vec3f Q11(+half.x * 0.65,  half.y * 0.10, half.z * 0.00);
    vec3f Q12(+half.x * 0.75,  half.y * 0.05, half.z * 0.03);

    vec2f tris[] = {
        vec2f(T_Triangle(q, Q0, Q1, Q2), 0),
        vec2f(T_Triangle(q, Q0, Q3, Q4), 1),
        vec2f(T_Triangle(q, Q3, Q4, Q5), 1),
        vec2f(T_Triangle(q, Q3, Q5, Q6), 1),
        vec2f(T_Triangle(q, Q7, Q8, Q9), 2),
        vec2f(T_Triangle(q, Q7, Q6, Q9), 2),
        vec2f(T_Triangle(q, Q7, Q6, Q5), 2),
        vec2f(T_Triangle(q, Q4, Q12, Q10), 1),
        vec2f(T_Triangle(q, Q12, Q10, Q11), 0)
    };

    vec2f sdf(Infinity, -1);
    for(int i = 0; i < 9; i++){
        bool accept = id < 0 || id == tris[i].y;
        if(sdf.x > tris[i].x && accept){
            sdf = tris[i];
        }
    }

    return sdf.x;
}

inline bb_cpu_gpu Float T_OrigamiBird(vec3f point){
    vec3f half(5.0);
    vec3f p = point;
    vec3f q(p.x, p.y, Absf(p.z));
    vec3f Q0(-half.x * 0.25,  half.y * 0.88, half.z * 0.30);
    vec3f Q1(+half.x * 0.05,  half.y * 0.50, half.z * 0.13);
    vec3f Q2(-half.x * 0.15,  half.y * 0.05, half.z * 0.13);
    vec3f Q3(+half.x * 0.60, -half.y * 0.10, half.z * 0.08);
    vec3f Q4(+half.x * 0.50,  half.y * 0.02, half.z * 0.03);
    vec3f Q5(+half.x * 0.75,  half.y * 0.20, half.z * 0.00);
    vec3f Q6(+half.x * 0.85,  half.y * 0.00, half.z * 0.00);
    vec3f Q7(-half.x * 0.20, -half.y * 0.50, half.z * 0.03);
    vec3f Q8(-half.x * 0.85, -half.y * 0.90, half.z * 0.00);
    vec3f Q9(-half.x * 0.25, -half.y * 0.80, half.z * 0.00);

    vec2f tris[] = {
        vec2f(T_Triangle(q, Q0, Q1, Q2), 0),
        vec2f(T_Triangle(q, Q1, Q2, Q3), 1),
        vec2f(T_Triangle(q, Q3, Q4, Q5), 2),
        vec2f(T_Triangle(q, Q6, Q5, Q3), 1),
        vec2f(T_Triangle(q, Q2, Q3, Q7), 1),
        vec2f(T_Triangle(q, Q2, Q8, Q9), 2)
    };

    vec2f sdf(Infinity, -1);
    for(int i = 0; i < 6; i++){
        if(sdf.x > tris[i].x){
            sdf = tris[i];
        }
    }

    return sdf.x;
}

