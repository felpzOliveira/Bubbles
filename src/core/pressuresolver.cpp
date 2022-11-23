#include <pressuresolver.h>
#include <sstream>
#include <stdio.h>

GridIndexVector::GridIndexVector(){}

GridIndexVector::GridIndexVector(int i, int j) : width(i), height(j){}

GridIndexVector::~GridIndexVector(){}


GridIndex2 GridIndexVector::operator[](int i) {
    return _getUnflattenedIndex(_indices[i]);
}

void GridIndexVector::insert(std::vector<GridIndex2> &indices) {
    reserve(_indices.size() + indices.size());
    for (unsigned int i = 0; i < indices.size(); i++) {
        push_back(indices[i]);
    }
}

void GridIndexVector::insert(GridIndexVector &indices) {
    reserve(_indices.size() + indices.size());
    for (unsigned int i = 0; i < indices.size(); i++) {
        int flatidx = indices.getFlatIndex(i);
        _indices.push_back(flatidx);
    }
}

std::vector<GridIndex2> GridIndexVector::getVector() {
    std::vector<GridIndex2> vector;
    vector.reserve(size());

    for (unsigned int i = 0; i < size(); i++){
        vector.push_back((*this)[i]);
    }

    return vector;
}

void GridIndexVector::getVector(std::vector<GridIndex2> &vector) {
    vector.reserve(size());

    for (unsigned int i = 0; i < size(); i++){
        vector.push_back((*this)[i]);
    }
}

/***************************************************************************/
VectorXd::VectorXd() {
}

VectorXd::VectorXd(int size) : _vector(size, 0.0) {
}

VectorXd::VectorXd(int size, double fill) : _vector(size, fill) {
}

VectorXd::VectorXd(VectorXd &vector) {
    _vector.reserve(vector.size());
    for (unsigned int i = 0; i < vector.size(); i++) {
        _vector.push_back(vector[i]);
    }
}

VectorXd::~VectorXd() {
}

double VectorXd::operator[](int i) const {
    return _vector[i];
}

double& VectorXd::operator[](int i) {
    return _vector[i];
}

void VectorXd::fill(double fill) {
    for (unsigned int i = 0; i < _vector.size(); i++) {
        _vector[i] = fill;
    }
}

double VectorXd::dot(VectorXd &vector) {
    double sum = 0.0;
    for (unsigned int i = 0; i < _vector.size(); i++) {
        sum += _vector[i] * vector._vector[i];
    }

    return sum;
}

double VectorXd::absMaxCoeff() {
    double max = -std::numeric_limits<double>::infinity();
    for (unsigned int i = 0; i < _vector.size(); i++) {
        if (fabs(_vector[i]) > max) {
            max = fabs(_vector[i]);
        }
    }

    return max;
}

/***************************************************************************/

MatrixCoefficients::MatrixCoefficients() {
}

MatrixCoefficients::MatrixCoefficients(int size) : cells(size, MatrixCell()) {
}

MatrixCoefficients::~MatrixCoefficients() {
}

const MatrixCell MatrixCoefficients::operator[](int i) const {
    return cells[i];
}

MatrixCell& MatrixCoefficients::operator[](int i) {
    return cells[i];
}
/***************************************************************************/

GridIndexKeyMap::GridIndexKeyMap() {
}

GridIndexKeyMap::GridIndexKeyMap(int i, int j) : _isize(i), _jsize(j){
    _indices = std::vector<int>(i*j, _notFoundValue);
}

GridIndexKeyMap::~GridIndexKeyMap() {
}

void GridIndexKeyMap::clear() {
    for (unsigned int i = 0; i < _indices.size(); i++) {
        _indices[i] = _notFoundValue;
    }
}

void GridIndexKeyMap::insert(GridIndex2 g, int key){
    insert(g.i, g.j, key);
}

void GridIndexKeyMap::insert(int i, int j, int key) {
    int flatidx = _getFlatIndex(i, j);
    _indices[flatidx] = key;
}

int GridIndexKeyMap::find(GridIndex2 g) {
    return find(g.i, g.j);
}

int GridIndexKeyMap::find(int i, int j){
    if (_indices.size() == 0) {
        return _notFoundValue;
    }

    if(i < 0 || i >= _isize || j < 0 || j >= _jsize)
        return _notFoundValue;

    int flatidx = _getFlatIndex(i, j);
    return _indices[flatidx];
}

/***************************************************************************/

bool TmpPressureSolver::isCellSolid(int i, int j){
    if(i < 0 || j < 0) return true;
    if(i >= _materialGrid->nx || j >= _materialGrid->ny) return true;
    Material mat = _materialGrid->At(i, j);
    return mat.type == Solid;
}

bool TmpPressureSolver::isCellFluid(int i, int j){
    if(i < 0 || j < 0) return false;
    if(i >= _materialGrid->nx || j >= _materialGrid->ny) return false;
    Material mat = _materialGrid->At(i, j);
    return mat.type == Fluid;
}

TmpPressureSolver::TmpPressureSolver() {
}

TmpPressureSolver::~TmpPressureSolver() {
}

void TmpPressureSolver::solve(PressureSolverParameters params, VectorXd &pressure) {
    _initialize(params);

    pressure.fill(0.0);

    _initializeGridIndexKeyMap();

    VectorXd b(_matSize);
    _calculateNegativeDivergenceVector(b);
    if (b.absMaxCoeff() < _pressureSolveTolerance) {
        return;
    }

    MatrixCoefficients A(_matSize);
    _calculateMatrixCoefficients(A);

    VectorXd precon(_matSize);
    _calculatePreconditionerVector(A, precon);

    _solvePressureSystem(A, b, precon, pressure);
}

void TmpPressureSolver::_initialize(PressureSolverParameters params) {
    _isize = params.materialGrid->nx;
    _jsize = params.materialGrid->ny;
    _dx = params.cellwidth;
    _density = params.density;
    _deltaTime = params.deltaTime;
    _fluidCells = params.fluidCells;
    _materialGrid = params.materialGrid;
    _vField = params.velocityField;
    _matSize = (int)_fluidCells->size();
}

void TmpPressureSolver::_initializeGridIndexKeyMap() {
    _keymap = GridIndexKeyMap(_isize, _jsize);
    for (unsigned int idx = 0; idx < _fluidCells->size(); idx++) {
        _keymap.insert(_fluidCells->at(idx), idx);
    }
}

void TmpPressureSolver::_calculateNegativeDivergenceVector(VectorXd &b) {

    double scale = 1.0f / (float)_dx;
    for (unsigned int idx = 0; idx < _fluidCells->size(); idx++) {
        int i = _fluidCells->at(idx).i;
        int j = _fluidCells->at(idx).j;

        double value = -scale * (double)(_vField->U(i + 1, j) - _vField->U(i, j) +
                                        _vField->V(i, j + 1) - _vField->V(i, j));
        b[_GridToVectorIndex(i, j)] = value;
    }

    // No functionality for moving solid cells, so velocity is 0
    float usolid = 0.0;
    float vsolid = 0.0;
    for (unsigned int idx = 0; idx < _fluidCells->size(); idx++) {
        int i = _fluidCells->at(idx).i;
        int j = _fluidCells->at(idx).j;
        int vidx = _GridToVectorIndex(i, j);

        if (isCellSolid(i-1, j)){
            b[vidx] -= (float)scale*(_vField->U(i, j) - usolid);
        }
        if (isCellSolid(i+1, j)) {
            b[vidx] += (float)scale*(_vField->U(i+1, j) - usolid);
        }

        if (isCellSolid(i, j-1)) {
            b[vidx] -= (float)scale*(_vField->V(i, j) - vsolid);
        }
        if (isCellSolid(i, j+1)){
            b[vidx] += (float)scale*(_vField->V(i, j+1) - vsolid);
        }
    }
}

int TmpPressureSolver::_getNumFluidOrAirCellNeighbours(int i, int j) {
    int n = 0;
    if (!isCellSolid(i-1, j)) { n++; }
    if (!isCellSolid(i+1, j)) { n++; }
    if (!isCellSolid(i, j-1)) { n++; }
    if (!isCellSolid(i, j+1)) { n++; }

    return n;
}

void TmpPressureSolver::_calculateMatrixCoefficients(MatrixCoefficients &A) {
    for (unsigned int idx = 0; idx < _fluidCells->size(); idx++) {
        int i = _fluidCells->at(idx).i;
        int j = _fluidCells->at(idx).j;
        int vidx = _GridToVectorIndex(i, j);

        int n = _getNumFluidOrAirCellNeighbours(i, j);
        A.cells[vidx].diag = (char)n;

        if (isCellFluid(i + 1, j)) {
            A.cells[vidx].plusi = 0x01;
        }

        if (isCellFluid(i, j + 1)) {
            A.cells[vidx].plusj = 0x01;
        }
    }
}

void TmpPressureSolver::_calculatePreconditionerVector(MatrixCoefficients &A, VectorXd &precon) {
    double scale = _deltaTime / (_density*_dx*_dx);
    double negscale = -scale;

    double tau = 0.97;      // Tuning constant
    double sigma = 0.25;    // safety constant
    for (unsigned int idx = 0; idx < _fluidCells->size(); idx++) {
        int i = _fluidCells->at(idx).i;
        int j = _fluidCells->at(idx).j;
        int vidx = _GridToVectorIndex(i, j);

        int vidx_im1 = _keymap.find(i - 1, j);
        int vidx_jm1 = _keymap.find(i, j - 1);

        double diag = (double)A[vidx].diag*scale;

        double plusi_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].plusi * negscale : 0.0;
        double plusi_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].plusi * negscale : 0.0;

        double plusj_im1 = vidx_im1 != -1 ? (double)A[vidx_im1].plusj * negscale : 0.0;
        double plusj_jm1 = vidx_jm1 != -1 ? (double)A[vidx_jm1].plusj * negscale : 0.0;

        double precon_im1 = vidx_im1 != -1 ? precon[vidx_im1] : 0.0;
        double precon_jm1 = vidx_jm1 != -1 ? precon[vidx_jm1] : 0.0;

        double v1 = plusi_im1 * precon_im1;
        double v2 = plusj_jm1 * precon_jm1;
        double v4 = precon_im1 * precon_im1;
        double v5 = precon_jm1 * precon_jm1;

        double e = diag - v1*v1 - v2*v2 - tau*(plusi_im1*(plusj_im1)*v4 +
                   plusj_jm1*(plusi_jm1)*v5);

        if (e < sigma*diag){
            e = diag;
        }

        if(fabs(e) > 10e-9){
            precon[vidx] = 1.0 / sqrt(e);
        }
    }
}

void TmpPressureSolver::_applyPreconditioner(MatrixCoefficients &A,
                        VectorXd &precon, VectorXd &residual, VectorXd &vect) {

    double scale = _deltaTime / (_density*_dx*_dx);
    double negscale = -scale;

    // Solve A*q = residual
    VectorXd q(_matSize);
    for (unsigned int idx = 0; idx < _fluidCells->size(); idx++) {
        int i = _fluidCells->at(idx).i;
        int j = _fluidCells->at(idx).j;
        int vidx = _GridToVectorIndex(i, j);

        int vidx_im1 = _keymap.find(i - 1, j);
        int vidx_jm1 = _keymap.find(i, j - 1);

        double plusi_im1 = 0.0;
        double precon_im1 = 0.0;
        double q_im1 = 0.0;
        if (vidx_im1 != -1) {
            plusi_im1  = (double)A[vidx_im1].plusi * negscale;
            precon_im1 = precon[vidx_im1];
            q_im1      = q[vidx_im1];
        }

        double plusj_jm1 = 0.0;
        double precon_jm1 = 0.0;
        double q_jm1 = 0.0;
        if (vidx_jm1 != -1) {
            plusj_jm1  = (double)A[vidx_jm1].plusj * negscale;
            precon_jm1 = precon[vidx_jm1];
            q_jm1      = q[vidx_jm1];
        }

        double t = residual[vidx] - plusi_im1 * precon_im1 * q_im1 -
        plusj_jm1 * precon_jm1 * q_jm1;

        t = t*precon[vidx];
        q[vidx] = t;
    }

    // Solve transpose(A)*z = q
    for (int idx = (int)_fluidCells->size() - 1; idx >= 0; idx--) {
        int i = _fluidCells->at(idx).i;
        int j = _fluidCells->at(idx).j;

        int vidx = _GridToVectorIndex(i, j);

        int vidx_ip1 = _keymap.find(i + 1, j);
        int vidx_jp1 = _keymap.find(i, j + 1);

        double vect_ip1 = vidx_ip1 != -1 ? vect[vidx_ip1] : 0.0;
        double vect_jp1 = vidx_jp1 != -1 ? vect[vidx_jp1] : 0.0;

        double plusi = (double)A[vidx].plusi * negscale;
        double plusj = (double)A[vidx].plusj * negscale;

        double preconval = precon[vidx];
        double t = q[vidx] - plusi * preconval * vect_ip1 -
                             plusj * preconval * vect_jp1;

        t = t*preconval;
        vect[vidx] = t;
    }
}

void TmpPressureSolver::_applyMatrix(MatrixCoefficients &A, VectorXd &x, VectorXd &result) {
    double scale = _deltaTime / (_density*_dx*_dx);
    double negscale = -scale;

    for (unsigned int idx = 0; idx < _fluidCells->size(); idx++) {
        int i = _fluidCells->at(idx).i;
        int j = _fluidCells->at(idx).j;

        // val = dot product of column vector x and idxth row of matrix A
        double val = 0.0;
        int vidx = _GridToVectorIndex(i - 1, j);
        if (vidx != -1) { val += x._vector[vidx]; }

        vidx = _GridToVectorIndex(i + 1, j);
        if (vidx != -1) { val += x._vector[vidx]; }

        vidx = _GridToVectorIndex(i, j - 1);
        if (vidx != -1) { val += x._vector[vidx]; }

        vidx = _GridToVectorIndex(i, j + 1);
        if (vidx != -1) { val += x._vector[vidx]; }

        val *= negscale;

        vidx = _GridToVectorIndex(i, j);
        val += (double)A.cells[vidx].diag * scale * x._vector[vidx];

        result._vector[vidx] = val;
    }
}

// v1 += v2*scale
void TmpPressureSolver::_addScaledVector(VectorXd &v1, VectorXd &v2, double scale) {
    for (unsigned int idx = 0; idx < v1.size(); idx++) {
        v1._vector[idx] += v2._vector[idx]*scale;
    }
}

// result = v1*s1 + v2*s2
void TmpPressureSolver::_addScaledVectors(VectorXd &v1, double s1,
                                    VectorXd &v2, double s2, VectorXd &result)
{
    for (unsigned int idx = 0; idx < v1.size(); idx++) {
        result._vector[idx] = v1._vector[idx]*s1 + v2._vector[idx]*s2;
    }
}

// Solve (A*pressure = b) with Modified Incomplete Cholesky
// Conjugate Gradient method (MICCG(0))
void TmpPressureSolver::_solvePressureSystem(MatrixCoefficients &A,
                                            VectorXd &b, VectorXd &precon,
                                            VectorXd &pressure) {

    double tol = _pressureSolveTolerance;
    if (b.absMaxCoeff() < tol) {
        return;
    }

    VectorXd residual(b);
    VectorXd auxillary(_matSize);
    _applyPreconditioner(A, precon, residual, auxillary);

    VectorXd search(auxillary);

    double alpha = 0.0;
    double beta = 0.0;
    double sigma = auxillary.dot(residual);
    double sigmaNew = 0.0;
    int iterationNumber = 0;

    while(iterationNumber < _maxCGIterations){
        _applyMatrix(A, search, auxillary);
        alpha = sigma / auxillary.dot(search);
        _addScaledVector(pressure, search, alpha);
        _addScaledVector(residual, auxillary, -alpha);

        if (residual.absMaxCoeff() < tol){
            printf("CG Iterations: %d: %g\n", iterationNumber, residual.absMaxCoeff());
            return;
        }

        _applyPreconditioner(A, precon, residual, auxillary);
        sigmaNew = auxillary.dot(residual);
        beta = sigmaNew / sigma;
        _addScaledVectors(auxillary, 1.0, search, beta, search);
        sigma = sigmaNew;

        iterationNumber++;

        if (iterationNumber % 10 == 0) {
            std::ostringstream ss;
            ss << "\tIteration #: " << iterationNumber <<
                        "\tEstimated Error: " << residual.absMaxCoeff() << std::endl;
            printf("%s", ss.str().c_str());
        }
    }

    printf("Iterations limit reached.\t Estimated error : %g\n",
    residual.absMaxCoeff());
}
