#include <tests.h>
#include <ctime>
#include <kernel.h>
#include <graphy.h>
#include <unistd.h>

static bool IsClose(Float a){ return Absf(a) < 1e-3; }

static int with_graphy = 1;

/*
* NOTE: How to do Monte-Carlo integration over a domain:
*       Step 1 - Find the Hypercube that contains the domain,
*                this function integrates over the unit circle
*                so the hypercube is the square (-1,-1) x (1, 1) -> Lrect = 2;
*       Step 2 - Find the area of the surface we want to integrate
*                the circle is Pi*R*R, however we can also let monte-carlo
*                compute it by the fraction of samples on the hypercube 
*                that falls on the domain, S = Pi*R*R ~ L*L*Nr/N = Vr;
*       Step 3 - Generate samples in the hypercube, for this function
*                it needs random x and y values, so we generate u1 and u2
*                uniformly inside the hypercube, if u1 and u2 are not
*                uniform than a PDF must be provided;
*       Step 4 - If a sample is inside our domain compute the function f
*                to be integrated and accumulate its result;
*       Step 5 - After many samples the average is: acc * Vr / Nr.
*                
*/
template<typename Q> Float MonteCarloOverUnitCircle(Q func_eval, long samples){
    Float acc = 0;
    Float L = 2;
    Float Nr = 0;
    Float N = 0;
    for(int s = 0; s < samples; s++){
        Float u1 = 2.f * rand_float() - 1.f;
        Float u2 = 2.f * rand_float() - 1.f;
        Float d = u1 * u1 + u2 * u2;
        Float r = d - 1.0;
        if(IsZero(r) || r < 0){
            acc += func_eval(sqrt(d));
            Nr += 1;
        }
        
        N += 1;
    }
    
    return (acc * (L * L * Nr / N) / Nr);
}

/*
* Same as above, but now everything is a volume.
*/
template<typename Q> Float MonteCarloOverUnitSphere(Q func_eval, long samples){
    Float acc = 0;
    Float L = 2;
    Float Nr = 0;
    Float N = 0;
    for(int s = 0; s < samples; s++){
        Float u1 = 2.f * rand_float() - 1.f;
        Float u2 = 2.f * rand_float() - 1.f;
        Float u3 = 2.f * rand_float() - 1.f;
        Float d = u1 * u1 + u2 * u2 + u3 * u3;
        Float r = d - 1.0;
        if(IsZero(r) || r < 0){
            acc += func_eval(sqrt(d));
            Nr += 1;
        }
        
        N += 1;
    }
    
    return (acc * (L * L * L * Nr / N) / Nr);
}

void test_kernels_2D(){
    printf("===== Test SPH Kernels 2D\n");
    long samples = 32768 * 2;
    long derivatives_count = 1024;
    SphStdKernel2 stdKernel(1.f);
    SphSpikyKernel2 spikyKernel(1.f);
    auto stdEval = [&](Float x) -> Float{ return stdKernel.W(x); };
    auto spikyEval = [&](Float x) -> Float{ return spikyKernel.W(x); };
    
    Float stdInt11 = MonteCarloOverUnitCircle(stdEval, samples);
    printf(" * Monte Carlo Standard Kernel integral: %g\n", stdInt11);
    stdInt11 = Absf(stdInt11 - 1.0);
    TEST_CHECK(stdInt11 < 0.05, "Kernel Std did not sum to 1");
    
    Float spikyInt11 = MonteCarloOverUnitCircle(spikyEval, samples);
    printf(" * Monte Carlo Spiky Kernel integral: %g\n", spikyInt11);
    spikyInt11 = Absf(spikyInt11 - 1.f);
    TEST_CHECK(spikyInt11 < 0.05, "Kernel Spiky did not sum to 1");
    
    Float vstd = stdKernel.W(0);
    Float vspk = spikyKernel.W(0);
    printf(" * Std W(0) = %g\n", vstd);
    printf(" * Spiky W(0) = %g\n", vspk);
    
    // Compare derivatives with finite differences
    Float e = 0.01;
    Float rad = 10.f;
    stdKernel.SetRadius(rad);
    spikyKernel.SetRadius(rad);
    for(int i = 0; i < derivatives_count; i++){
        Float x, y, d;
        do{
            x = (2.f * rand_float() - 1.f) * rad;
            y = (2.f * rand_float() - 1.f) * rad;
            d = sqrt(x * x + y * y);
        }while(d > rad - e);
        
        Float finite_dW = (stdKernel.W(d + e) - stdKernel.W(d - e)) / (2.0 * e);
        
        Float finite_d2W = (stdKernel.W(d + e) - 2.0 * stdKernel.W(d) + 
                            stdKernel.W(d - e)) / (e * e);
        
        Float finite_dW2 = (spikyKernel.W(d + e) - spikyKernel.W(d - e)) / (2.0 * e);
        
        Float finite_d2W2 = (spikyKernel.W(d + e) - 2.0 * spikyKernel.W(d) + 
                             spikyKernel.W(d - e)) / (e * e);
        
        Float dW  = stdKernel.dW(d);
        Float d2W = stdKernel.d2W(d);
        
        Float dW2  = spikyKernel.dW(d);
        Float d2W2 = spikyKernel.d2W(d);
        
        Float v = finite_dW - dW;
        Float v2 = finite_dW2 - dW2;
        Float v3 = finite_d2W - d2W;
        Float v4 = finite_d2W2 - d2W2;
        
        if(!IsClose(v)){
            printf("Failed std dW : %g != %g (%g) [%d]\n", dW, finite_dW, v, i);
        }
        
        if(!IsClose(v2)){
            printf("Failed spiky dW : %g != %g (%g) [%d]\n", dW2, finite_dW2, v2, i);
        }
        
        if(!IsClose(v3)){
            printf("Failed std d2W : %g != %g (%g) [%d]\n", d2W, finite_d2W, v3, i);
        }
        
        if(!IsClose(v4)){
            printf("Failed spiky d2W : %g != %g (%g) [%d]\n", d2W2, finite_d2W2, v4, i);
        }
        
        TEST_CHECK(IsClose(v) && IsClose(v2) && IsClose(v3) && IsClose(v4), 
                   "Invalid derivative for kernel");
    }
    
    printf(" * Derivatives passed ( # %d )\n", (int)derivatives_count);
    if(with_graphy){
        int points = 60;
        float *p = new float[points * 3];
        Float rad = 1.0;
        stdKernel.SetRadius(rad);
        Float dx = 2.0*rad / (Float)points;
        Float xmin = -rad;
        int it = 0;
        Float maxh = 0.f;
        for(int pi = 0; pi < points; pi++){
            Float x = xmin + pi * dx;
            Float y = stdKernel.W(x);
            maxh = y > maxh ? y : maxh;
            p[it + 0] = x;
            p[it + 1] = y - 0.5;
            p[it + 2] = 0;
            it += 3;
        }
        
        float rgb[3] = {1,0,0};
        graphy_render_points(p, rgb, points, -rad, rad, maxh, -maxh);
        
        sleep(2.0);
        
        delete[] p;
    }
    
    printf("===== OK\n");
}

void test_kernels_3D(){
    printf("===== Test SPH Kernels 3D\n");
    long samples = 32768 * 2;
    long derivatives_count = 1024;
    SphStdKernel3 stdKernel(1.f);
    SphSpikyKernel3 spikyKernel(1.f);
    auto stdEval = [&](Float x) -> Float{ return stdKernel.W(x); };
    auto spikyEval = [&](Float x) -> Float{ return spikyKernel.W(x); };
    
    Float stdInt11 = MonteCarloOverUnitSphere(stdEval, samples);
    printf(" * Monte Carlo Standard Kernel integral: %g\n", stdInt11);
    stdInt11 = Absf(stdInt11 - 1.0);
    TEST_CHECK(stdInt11 < 0.05, "Kernel Std did not sum to 1");
    
    Float spikyInt11 = MonteCarloOverUnitSphere(spikyEval, samples);
    printf(" * Monte Carlo Spiky Kernel integral: %g\n", spikyInt11);
    spikyInt11 = Absf(spikyInt11 - 1.f);
    TEST_CHECK(spikyInt11 < 0.05, "Kernel Spiky did not sum to 1");
    
    Float vstd = stdKernel.W(0);
    Float vspk = spikyKernel.W(0);
    printf(" * Std W(0) = %g\n", vstd);
    printf(" * Spiky W(0) = %g\n", vspk);
    
    // Compare derivatives with finite differences
    Float e = 0.01;
    Float rad = 10.f;
    stdKernel.SetRadius(rad);
    spikyKernel.SetRadius(rad);
    for(int i = 0; i < derivatives_count; i++){
        Float x, y, z, d;
        do{
            x = (2.f * rand_float() - 1.f) * rad;
            y = (2.f * rand_float() - 1.f) * rad;
            z= (2.f * rand_float() - 1.f) * rad;
            d = sqrt(x * x + y * y + z * z);
        }while(d > rad - e);
        
        Float finite_dW = (stdKernel.W(d + e) - stdKernel.W(d - e)) / (2.0 * e);
        
        Float finite_d2W = (stdKernel.W(d + e) - 2.0 * stdKernel.W(d) + 
                            stdKernel.W(d - e)) / (e * e);
        
        Float finite_dW2 = (spikyKernel.W(d + e) - spikyKernel.W(d - e)) / (2.0 * e);
        
        Float finite_d2W2 = (spikyKernel.W(d + e) - 2.0 * spikyKernel.W(d) + 
                             spikyKernel.W(d - e)) / (e * e);
        
        Float dW  = stdKernel.dW(d);
        Float d2W = stdKernel.d2W(d);
        
        Float dW2  = spikyKernel.dW(d);
        Float d2W2 = spikyKernel.d2W(d);
        
        Float v = finite_dW - dW;
        Float v2 = finite_dW2 - dW2;
        Float v3 = finite_d2W - d2W;
        Float v4 = finite_d2W2 - d2W2;
        
        if(!IsClose(v)){
            printf("Failed std dW : %g != %g (%g) [%d]\n", dW, finite_dW, v, i);
        }
        
        if(!IsClose(v2)){
            printf("Failed spiky dW : %g != %g (%g) [%d]\n", dW2, finite_dW2, v2, i);
        }
        
        if(!IsClose(v3)){
            printf("Failed std d2W : %g != %g (%g) [%d]\n", d2W, finite_d2W, v3, i);
        }
        
        if(!IsClose(v4)){
            printf("Failed spiky d2W : %g != %g (%g) [%d]\n", d2W2, finite_d2W2, v4, i);
        }
        
        TEST_CHECK(IsClose(v) && IsClose(v2) && IsClose(v3) && IsClose(v4), 
                   "Invalid derivative for kernel");
    }
    
    // graphy does not support 3D rendering for now
    printf(" * Derivatives passed ( # %d )\n", (int)derivatives_count);
    printf("===== OK\n");
}