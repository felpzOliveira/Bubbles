#include <geometry.h>
#include <graphy-inl.h>
#include <vgrid.h>
#include <iostream>
#include <sstream>

/*********************************************************************************/
#define I2(i, j, nx) ((i) + (j) * (nx))

template<typename T>
__bidevice__ T lerp(T left, T right, Float dx){
    return left + dx * (right - left);
}

template<typename T>
__bidevice__ T bilerp(T f00, T f10, T f01, T f11, Float dx, Float dy){
    return lerp(lerp(f00, f10, dx), lerp(f01, f11, dx), dy);
}

template<typename T>
__bidevice__ T qf_near_value(T *qf, Float u, Float v, int nx, int ny){
    int i = Max(0, Min((int)u, nx-1));
    int j = Max(0, Min((int)v, ny-1));
    return qf[I2(i, j, nx)];
}

template<typename T>
__bidevice__ T qf_sample_value(T *qf, vec2f p, int nx, int ny){
    Float s = p.x - 0.5,
          t = p.y - 0.5;

    int iu = (int)s, iv = (int)t;
    Float fu = s - iu, fv = t - iv;
    T f00 = qf_near_value(qf, iu + 0.5, iv + 0.5, nx, ny);
    T f10 = qf_near_value(qf, iu + 1.5, iv + 0.5, nx, ny);
    T f01 = qf_near_value(qf, iu + 0.5, iv + 1.5, nx, ny);
    T f11 = qf_near_value(qf, iu + 1.5, iv + 1.5, nx, ny);
    return bilerp(f00, f10, f01, f11, fu, fv);
}

template<typename T>
__bidevice__ void advect_rk3(vec2f *vf, T *qf, T *new_qf, Float dt,
                             int i, int j, int nx, int ny)
{
    vec2f p = vec2f(i + 0.5, j + 0.5);
    vec2f k1 = qf_sample_value(vf, p, nx, ny);
    vec2f k2 = qf_sample_value(vf, p - 0.50 * dt * k1, nx, ny);
    vec2f k3 = qf_sample_value(vf, p - 0.75 * dt * k2, nx, ny);
    vec2f p_prev = p - (dt / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3);
    new_qf[I2(i, j, nx)] = qf_sample_value(qf, p_prev, nx, ny);
}

template<typename T>
__bidevice__ void addInflow(T *qf, vec4f rect, T value, int i, int j, int nx){
    Float lower_x = rect.x,
          lower_y = rect.y,
          upper_x = rect.z,
          upper_y = rect.w;

    if(i >= lower_x && i <= upper_x && j >= lower_y && j <= upper_y){
        qf[I2(i, j, nx)] = value;
    }
}

template<typename T>
class DoubleBuffer{
    public:
    T *cur;
    T *nxt;
    int nx;
    int ny;

    __bidevice__ DoubleBuffer(){nx = 0, ny = 0; cur = nullptr; nxt = nullptr;}

    void build(int _nx, int _ny){
        size_t total = _nx * _ny;
        cur = cudaAllocateVx(T, total);
        nxt = cudaAllocateVx(T, total);
        nx = _nx;
        ny = _ny;
        for(size_t i = 0; i < total; i++){
            cur[i] = T(0);
            nxt[i] = T(0);
        }
    }

    void flip(){
        T *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    void advect(vec2f *vf, Float dt){
        int res_x = nx;
        int res_y = ny;
        size_t items = nx * ny;
        AutoParallelFor("advect", items, AutoLambda(size_t index){
            int iy = index / res_x;
            int ix = index % res_x;
            advect_rk3(vf, cur, nxt, dt, ix, iy, res_x, res_y);
        });
    }

    void set_inflow(vec4f rect, T value){
        int res_x = nx;
        size_t items = nx * ny;
        AutoParallelFor("inflow", items, AutoLambda(size_t index){
            int iy = index / res_x;
            int ix = index % res_x;
            addInflow(cur, rect, value, ix, iy, res_x);
        });
    }

    std::string ToString(int w, int h, int n_w, bool showCur=true) const{
        std::stringstream ss;
        T *ptr = showCur ? cur : nxt;
        ss << " - Width: " << nx << std::endl;
        ss << " - Height: " << ny << std::endl;
        ss << " - Data: " << std::endl;
        ss << std::left << std::setw(n_w) << std::setfill(' ') << " ";
        for(int j = 1; j < nx+1; j++){
            std::string num("(");
            num += std::to_string(j);
            num += ")";
            ss << std::left << std::setw(n_w) << std::setfill(' ') << num;
            if(j == w && j < nx-1){
                ss << std::left << std::setw(n_w) << std::setfill(' ') << "...";
                j = nx-1;
            }
        }

        ss << std::endl;
        for(int j = 0; j < ny; j++){
            std::string num("(");
            num += std::to_string(j+1);
            num += ")";
            ss << std::left << std::setw(n_w) << std::setfill(' ') << num;
            for(int i = 0; i < nx; i++){
                if(i < w || i == nx-1){
                    ss << std::left << std::setw(n_w) << std::setfill(' ') <<
                    std::fixed << std::setprecision(4) << ptr[I2(i, j, nx)];
                }else if(i == w && i < nx-1){
                    ss << std::left << std::setw(n_w) << std::setfill(' ') << "...";
                    i = nx-2;
                }
            }

            if(j < ny-1)
                ss << std::endl;

            if(j == h && j < ny-1){
                ss << "..." << std::endl;
                j = ny-2;
            }
        }
        return ss.str();
    }
};

template<typename T>
inline std::ostream &operator<<(std::ostream &out, const DoubleBuffer<T> &buf){
    const int w = 8;
    const int h = 6;
    const int n_w = 8;
    out << buf.ToString(w, h, n_w);
    return out;
}

typedef DoubleBuffer<Float> DoubleBuffer1f;
typedef DoubleBuffer<vec2f> DoubleBuffer2f;
typedef DoubleBuffer<vec3f> DoubleBuffer3f;

template<typename T>
__bidevice__ T buf_at(T *buf, int i, int j, int nx, int ny, T def){
    if(i >= nx || j >= ny || i < 0 || j < 0) return def;
    return buf[I2(i, j, nx)];
}


__bidevice__ vec2f vel_at(vec2f *vf, int i, int j, int nx, int ny){
    return buf_at(vf, i, j, nx, ny, vec2f(0.f));
}

__bidevice__ Float p_at(Float *pf, int i, int j, int nx, int ny){
    return buf_at(pf, i, j, nx, ny, 0.f);
}


__bidevice__ vec2f vel_with_boundary(vec2f *vf, int i, int j, int nx, int ny){
    if((i == 0 && j == 0) || (i == nx-1 && j == ny-1) ||
       (i == 0 && j == ny-1) || (i == nx-1 && j == 0))
    {
        vf[I2(i, j, nx)] = vec2f(0.f);
    }else if(i == 0){
        vf[I2(i, j, nx)] = -vf[I2(i+1, j, nx)];
    }else if(j == 0){
        vf[I2(i, j, nx)] = -vf[I2(i, j+1, nx)];
    }else if(i == nx-1){
        vf[I2(i, j, nx)] = -vf[I2(i-1, j, nx)];
    }else if(j == ny-1){
        vf[I2(i, j, nx)] = -vf[I2(i, j-1, nx)];
    }

    return vf[I2(i, j, nx)];
}

__bidevice__ Float p_with_boundary(Float *pf, int i, int j, int nx, int ny){
    if(i >= nx || j >= ny || i < 0 || j < 0) return 0.f;
    else if((i == 0 && j == 0) || (i == nx-1 && j == ny-1) ||
       (i == 0 && j == ny-1) || (i == nx-1 && j == 0))
    {
        pf[I2(i, j, nx)] = 0.f;
    }else if(i == 0){
        pf[I2(i, j, nx)] = pf[I2(i+1, j, nx)];
    }else if(j == 0){
        pf[I2(i, j, nx)] = pf[I2(i, j+1, nx)];
    }else if(i == nx-1){
        pf[I2(i, j, nx)] = pf[I2(i-1, j, nx)];
    }else if(j == ny-1){
        pf[I2(i, j, nx)] = pf[I2(i, j-1, nx)];
    }

    return pf[I2(i, j, nx)];
}

__host__ void apply_vel_bc(vec2f *vf, int nx, int ny){
    size_t items = nx * ny;
    AutoParallelFor("vel bound", items, AutoLambda(size_t index){
        int iy = index / nx;
        int ix = index % nx;
        vel_with_boundary(vf, ix, iy, nx, ny);
    });
}

__host__ void apply_p_bc(Float *pf, int nx, int ny){
    size_t items = nx * ny;
    AutoParallelFor("p bound", items, AutoLambda(size_t index){
        int iy = index / nx;
        int ix = index % nx;
        p_with_boundary(pf, ix, iy, nx, ny);
    });
}

__host__ Float pressure_jacobi_iter(Float *pf, Float *pf_new, Float *divf,
                                    int nx, int ny, Float *norm_new, Float *norm_dif)
{
    size_t items = nx * ny;
    *norm_new = 0;
    *norm_dif = 0;

    AutoParallelFor("jacobi", items, AutoLambda(size_t index){
        int j = index / nx;
        int i = index % nx;
        pf_new[I2(i, j, nx)] = 0.25 * (p_with_boundary(pf,i+1,j,nx,ny) +
                                       p_with_boundary(pf,i-1,j,nx,ny) +
                                       p_with_boundary(pf,i,j+1,nx,ny) +
                                       p_with_boundary(pf,i,j-1,nx,ny) - divf[I2(i,j,nx)]);

        Float pf_new_ij = pf_new[I2(i, j, nx)];
        Float pf_diff = Absf(pf_new_ij - p_with_boundary(pf, i, j, nx, ny));
        atomicAdd(norm_new, (pf_new_ij * pf_new_ij));
        atomicAdd(norm_dif, pf_diff * pf_diff);
    });

    Float residual = sqrt(*norm_dif / *norm_new);
    if(*norm_new == 0) residual = 0.f;
    return residual;
}

__host__ void pressure_jacobi(DoubleBuffer1f *pressures, Float *divf,
                              Float *norm_new, Float *norm_diff)
{
    Float residual = 10.f;
    int counter = 0;
    int nx = pressures->nx;
    int ny = pressures->ny;
    while(residual > 0.001){
        Float *pf_curr = pressures->cur;
        Float *pf_next = pressures->nxt;
        residual = pressure_jacobi_iter(pf_curr, pf_next, divf, nx,
                                        ny, norm_new, norm_diff);
        pressures->flip();
        counter ++;
        if(counter > 2) break;
        printf("Erro = %g\n", residual);
    }

    apply_p_bc(pressures->cur, nx, ny);
}

__bidevice__ Float _divergence(vec2f *field, Float *divf, int i, int j, int nx, int ny){
    Float xp = vel_at(field, i+1,j,nx,ny).x;
    Float xn = vel_at(field, i-1,j,nx,ny).x;
    Float yp = vel_at(field, i,j+1,nx,ny).y;
    Float yn = vel_at(field, i,j-1,nx,ny).y;
    Float div = 0.5 * ((xp - xn) + (yp - yn));
    divf[I2(i, j, nx)] = div;
    return div;
}

__host__ void divergence(vec2f *field, Float *divf, int nx, int ny){
    size_t items = nx * ny;
    AutoParallelFor("Div", items, AutoLambda(size_t index){
        int j = index / nx;
        int i = index % nx;
        _divergence(field, divf, i, j, nx, ny);
    });
}

__host__ void correct_divergence(vec2f *vf, vec2f *vf_new, Float *pf, int nx, int ny){
    size_t items = nx * ny;
    AutoParallelFor("div2", items, AutoLambda(size_t index){
        int j = index / nx;
        int i = index % nx;
        Float px = (p_at(pf,i+1,j,nx,ny) - p_at(pf,i-1,j,nx,ny)) * 0.5;
        Float py = (p_at(pf,i,j+1,nx,ny) - p_at(pf,i,j-1,nx,ny)) * 0.5;
        vf_new[I2(i,j,nx)] = vf[I2(i,j,nx)] - vec2f(px, py);
    });
}

void test_routine(){
    int res = 800;
    size_t total = res * res;
    DoubleBuffer2f *velocities = cudaAllocateVx(DoubleBuffer2f, 1);
    DoubleBuffer1f *pressures = cudaAllocateVx(DoubleBuffer1f, 1);
    DoubleBuffer3f *densities = cudaAllocateVx(DoubleBuffer3f, 1);

    velocities->build(res, res);
    pressures->build(res, res);
    densities->build(res, res);

    Float *velocity_divs = cudaAllocateVx(Float, total);
    Float *diff_pressures = cudaAllocateVx(Float, total);

    Float *norm_new = cudaAllocateVx(Float, 1);
    Float *norm_diff = cudaAllocateVx(Float, 1);
    int pixel_mid = (int)(res / 2.0);
    int ix_length = 15 * res / 512;
    int iy_length = 10;
    vec4f area = vec4f(pixel_mid - ix_length, 8, pixel_mid + ix_length, 8 + iy_length);
    vec2f inflow_velocity = vec2f(0.0, 3.0);
    vec3f temp_dye = vec3f(0.5, 0.7, 0.5);
    Float dt = 0.05;

    GWindow gui("MAC Test");
    auto canvas = gui.get_canvas();
    canvas.Color(0x112F41);

    int _w = res, _h = res;

    auto fetcher = [&](int x, int y) -> GVec4f{
        int f = x + y * res;
        //vec2f val = velocities->cur[f] * 0.5 + vec2f(0.5);
        vec3f val = densities->cur[f];
        return GVec4f(Absf(val.x), Absf(val.y), Absf(val.z), 1.f);
    };

    while(true){
        for(int i = 0; i < 15; i++){
            velocities->set_inflow(area, inflow_velocity);
            densities->set_inflow(area, temp_dye);

            apply_vel_bc(velocities->cur, res, res);

            velocities->advect(velocities->cur, dt);
            densities->advect(velocities->cur, dt);

            velocities->flip();
            densities->flip();

            apply_vel_bc(velocities->cur, res, res);

            divergence(velocities->cur, velocity_divs, res, res);
            pressure_jacobi(pressures, velocity_divs, norm_new, norm_diff);

            correct_divergence(velocities->cur, velocities->nxt, pressures->cur, res, res);
            velocities->flip();
        }

        canvas.for_each_pixel([&](int x, int y) -> GVec4f{
            return canvas.upsample_from(x, y, _w, _h, fetcher, GUpsampleMode::Bilinear);
        });
        gui.update();
        printf("?\n");
    }
}


/*********************************************************************************/

