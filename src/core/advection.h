/* date = November 16th 2022 10:25 */
#pragma once
#include <geometry.h>
#include <sampling.h>
#include <vgrid.h>
#include <mac_grid.h>

/* Simple forward Euler method for velocity integration in time */
template<typename Sampler, typename T>
struct EulerIntegrator{
    Float dt;
    T p;
    Float h;
    Sampler fn;

    bb_cpu_gpu T Backtrack(){
        T vel = fn(p) / h;
        return p - vel * dt;
    }

};

/* Third order Runge-Kutta method for velocity integration in time */
template<typename Sampler, typename T>
struct RK3Integrator{
    Float dt;
    T p;
    Float h;
    Sampler fn;

    bb_cpu_gpu T Backtrack(){
        T k1 = fn(p) / h;
        T k2 = fn(p - 0.50 * dt * k1) / h;
        T k3 = fn(p - 0.75 * dt * k2) / h;
        return p - (dt / 9.) * (2 * k1 + 3 * k2 + 4 * k3);
    }
};

template<typename T, typename MACVelocity, typename GridData, typename Interpolator>
void AdvectQuantity(MACVelocity *macGrid, GridData *quantity, Float interval)
{
    auto *u = macGrid->U_ptr();
    auto *v = macGrid->V_ptr();
    auto *w = macGrid->W_ptr();
    size_t items = quantity->Length();
    Float spacing = quantity->Spacing();

    AutoParallelFor("Advection", items, AutoLambda(size_t index){
        Interpolator interpolator;
        T pos = quantity->DataGridPosition(index);

        auto vel_at = [&](T p) -> T{
            T res(0);
            res[0] = u->SampleGridCoords(p.x, p.y, interpolator);
            res[1] = v->SampleGridCoords(p.x, p.y, interpolator);
            if(w) res[2] = w->SampleGridCoords(p.x, p.y, interpolator);
            return res;
        };

        using sampler = decltype(vel_at);
        RK3Integrator<sampler, T> integrator{interval, pos, spacing, vel_at};
        T s = integrator.Backtrack();

        // TODO: Colider, either query sdf and move the backtrack point
        //       or extrapolate the values for quantity before calling this

        quantity->SetNext(index, quantity->SampleGridCoords(s.x, s.y, interpolator));
    });
}

class GridAdvectionSolver2{
    public:
    GridAdvectionSolver2() = default;
    ~GridAdvectionSolver2() = default;

    void Advect(MACVelocityGrid2 *macGrid, GridData2f *q, Float interval){
        AdvectQuantity<vec2f, MACVelocityGrid2, GridData2f, MonotonicCatmull>
            (macGrid, q, interval);
    }
};

