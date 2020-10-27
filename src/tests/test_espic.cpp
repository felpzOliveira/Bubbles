#include <tests.h>
#include <graphy.h>
#include <sph_solver.h>
#include <emitter.h>
#include <unistd.h>
#include <espic_solver.h>
#include <statics.h>

void SetPositionBuffer(SpecieSet2 **sets, int n, float *pos, float scale=1){
    int it = 0;
    for(int k = 1; k < n; k++){
        SpecieSet2 *pSet = sets[k];
        
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            vec2f p = pSet->GetParticlePosition(i);
            pos[3 * it + 0] = p.x * scale;
            pos[3 * it + 1] = p.y * scale;
            pos[3 * it + 2] = 0;
            it += 1;
        }
    }
}

void SetPositionColorBuffer(SpecieSet2 **sets, int n, float *pos, float *col, float scale=1){
    int it = 0;
    for(int k = 1; k < n; k++){
        SpecieSet2 *pSet = sets[k];
        Float rgb[3] = {1,0,0};
        if(k == 1){
            rgb[1] = 1;
            rgb[2] = 1;
        }
        
        for(int i = 0; i < pSet->GetParticleCount(); i++){
            vec2f p = pSet->GetParticlePosition(i);
            pos[3 * it + 0] = p.x * scale;
            pos[3 * it + 1] = p.y * scale;
            pos[3 * it + 2] = 0;
            
            col[3 * it + 0] = rgb[0];
            col[3 * it + 1] = rgb[1];
            col[3 * it + 2] = rgb[2];
            it += 1;
        }
        
    }
}

bool SolvePotentialGS(Float dx, Float *phi, Float *rho, int ni, int max_it = 5000){
    Float L2 = 0;
    Float dx2 = dx * dx;
    Float w = 1.4;
    Float EPS0 = PermittivityEPS;
    for(int solver_it = 0; solver_it < max_it; solver_it++){
        phi[0] = 0;
        phi[ni-1] = 0;
        
        for(int i = 1; i < ni - 1; i++){
            Float g = 0.5 * (phi[i-1] + phi[i+1] + dx2 * rho[i] / EPS0);
            phi[i] = phi[i]  + w * (g - phi[i]);
        }
        
        if(solver_it % 50 == 0){
            Float sum = 0;
            for(int i = 1; i < ni - 1; i++){
                Float R = -rho[i] / EPS0 - (phi[i-1] - 2 * phi[i] + phi[i+1]) / dx2;
                sum += R * R;
            }
            
            L2 = sqrt(sum / (Float)ni);
            if(L2 < 1e-6){
                // solved!
                return false;
            }
        }
    }
    
    return true;
}

void ComputeEF(Float dx, Float *ef, Float *phi, int ni, bool second_order = true){
    for(int i = 1; i < ni - 1; i++){
        ef[i] = -(phi[i+1] - phi[i-1]) / (2.0 * dx);
    }
    
    if(second_order){
        ef[0] = (3.0 * phi[0] - 4.0 * phi[1] + phi[2]) / (2.0 * dx);
        ef[ni-1] = (-phi[ni-3]+ 4.0 * phi[ni-2] - 3.0 * phi[ni-1])/(2.0 * dx);
    }else{
        ef[0] = (phi[0] - phi[1]) / dx;
        ef[ni-1] = (phi[ni-2] - phi[ni-1]) / dx;
    }
}

void test_espic_1D(){
    printf("===== ES-PIC 1D -- Potential Well\n");
    const int ni = 21; // nodes
    Float QE = ElementaryQE;
    
    Float phi[ni]; // potential
    Float rho[ni]; // charge density
    Float ef[ni]; // electric field
    
    Float x0 = 0;
    Float x1 = 0.6; // meters
    
    Float dx = (x1 - x0) / (ni - 1.0); // cell length
    
    
    for(int i = 0; i < ni; i++){ // initializes nodes from domain
        phi[i] = 0; ef[i] = 0;
        rho[i] = QE * 1e12;
    }
    
    /*
    * First Chapter: page 19
    * Poisson: L(phi) = -rho / e0, in this case rho is constant
    * rho = e n0, so by integrating this thing we write:
    * d(phi)/dx = -(rho/e0) x + A
    *       phi = -(rho/e0)x^2 / 2 + A x + B
    *
    * Boundary: phi = 0 for x = 0  ==> B = 0
    *           phi = 0 for x = x1 ==> A = rho x / (2 e0)
    *
    * phi = rho/(2 e0) x (x1 - x)
    *
    * For electric field E = -d(phi)/dx ==> E = (rho/e0) (x - x1/2)
    *
    * However for this to be usefull we use Finite Difference Method
    * remenber taylor series: A value of a function at f(x + dx) is approximated
    * by derivatives on point x:
    * f(x + dx) = f(x) + (dx/1!)d(f)/dx + ((dx)^2/2!)d^2(f)/dx^2 + ((dx)^3/3!)d^3(f)/dx^3 ...
    * by inverting the signs (+/-) we can also compute f(x - dx):
    *
    * f(x - dx) = f(x) - (dx/1!)d(f)/dx + ((dx)^2/2!)d^2(f)/dx^2 - ((dx)^3/3!)d^3(f)/dx^3 ...
    * This gives the second derivative approximation for central diferences:
    * d²f/dx² = (fi-1 - 2fi + fi+1)/(dx)² + O(3) (sum both equations)
    * Applying to Poisson:
    * (phii-1 - 2phii + phii+1)/(dx)² = -(rhoi/e0) for i[1, ni-2]
    * Meaning: the potential at node i depends on nodes i+1 and i-1 so for first and last
    * nodes we impose phi0 = phini-1 = 0
    * see matrix formulation.
    *
*/
    
    // parameters
    Float q = -QE; // particle charge
    Float m = ElectronMass; // particle mass
    Float x = 4 * dx; // initial position
    Float v = 0; // particle velocity
    Float dt = 1e-9; // timestep
    
    SolvePotentialGS(dx, phi, rho, ni); // compute potential at each cell
    
    ComputeEF(dx, ef, phi, ni); // compute electric field at each cell
    
    Float li = (x - x0) / dx; // initial index
    int id = (int)li;
    Float efp = Mix(ef[id], ef[id+1], li); // interpolate electric field
    
    // rewind particle
    v -= 0.5 * (q/m) * efp * dt;
    
    float pos[3] = {0, 0, 0};
    float col[3] = {1, 0, 0};
    
    Float minx = 0.11;
    Float maxx = 0.20;
    
    int runs = 4000;
    for(int ts = 0; ts < runs; ts++){
        li = (x - x0) / dx;
        id = (int)li;
        efp = Mix(ef[id], ef[id+1], li);
        
        v += (q/m) * efp * dt;
        x += v * dt;
        
        pos[0] = (x - minx) * (1.0 - (-1.0)) / (maxx - minx) + (-1.0);
        Debug_GraphyDisplayParticles(1, &pos[0], &col[0]);
    }
    
    printf("===== OK\n");
}

void test_espic_particles_2D(){
    printf("===== ES-PIC Particle distribution\n");
    ParticleSetBuilder2 pBuilderE, pBuilderI;
    
    EspicSolver2 solver;
    ColliderSetBuilder2 colliderBuilder;
    
    Shape2 *container = MakeRectangle2(Translate2(0,0), vec2f(0.2,0.2), true);
    colliderBuilder.AddCollider2(container);
    ColliderSet2 *collider = colliderBuilder.GetColliderSet();
    
    Float ME = ElectronMass;
    Float QE = ElementaryQE;
    Float MI = 16.0 * AtomicMassUnit;
    vec2f p0(-0.1,-0.1), p1(0.0,0.0);
    vec2f p01(-0.1,-0.1), p11(0.1,0.1);
    
    Bounds2f domainI(p01, p11);
    Bounds2f domainE(vec2f(-0.1, -0.1), vec2f(0.0, 0.0));
    
    UniformBoxParticleEmitter2 emitterE(domainE, vec2ui(21, 21));
    UniformBoxParticleEmitter2 emitterI(domainI, vec2ui(41, 41));
    
    emitterE.Emit(&pBuilderE, 1e11);
    emitterI.Emit(&pBuilderI, 1e11);
    
    int speciesCount = 2;
    SpecieSet2 **ppSet = cudaAllocateVx(SpecieSet2*, speciesCount);
    ppSet[0] = SpecieSet2FromBuilder(&pBuilderI, MI, QE, 1); // Ions
    ppSet[1] = SpecieSet2FromBuilder(&pBuilderE, ME, -QE, 0); // Electrons
    
    int total = ppSet[0]->GetParticleCount() + ppSet[1]->GetParticleCount();
    
    float *pos = new float[3 * total];
    float *col = new float[3 * total];
    
    SetPositionColorBuffer(ppSet, speciesCount, pos, col, 8);
    
    Grid2 *grid = MakeGrid(vec2ui(21,21), p01, p11);
    solver.Setup(grid, ppSet, speciesCount);
    solver.SetColliders(collider);
    
    for(int i = 0; i < 10000; i++){
        solver.Advance(2e-10);
        
        SetPositionBuffer(ppSet, speciesCount, pos, 8);
        Debug_GraphyDisplayParticles(ppSet[1]->GetParticleCount(), 
                                     &pos[0], &col[0], 5.0);
    }
    
    delete[] col;
    delete[] pos;
    printf("===== OK\n");
}
