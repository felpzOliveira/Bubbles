/* date = November 17th 2022 18:16 */
#pragma once
#include <vgrid.h>
#include <mac_grid.h>

#include <vector>
#include <stdio.h>
#include <iostream>
#include <vector>

class GridIndexVector
{
    public:
    GridIndexVector();
    GridIndexVector(int i, int j);
    ~GridIndexVector();

    inline size_t size() {
        return _indices.size();
    }

    inline bool empty() {
        return _indices.empty();
    }

    inline void reserve(size_t n) {
        _indices.reserve(n);
    }

    inline void shrink_to_fit() {
        _indices.shrink_to_fit();
    }

    GridIndex2 operator[](int i);

    inline GridIndex2 at(int i) {
        return _getUnflattenedIndex(_indices[i]);
    }

    inline GridIndex2 get(int i) {
        return _getUnflattenedIndex(_indices[i]);
    }

    inline unsigned int getFlatIndex(int i) {
        return _indices[i];
    }

    inline GridIndex2 front() {
        return _getUnflattenedIndex(_indices.front());
    }

    inline GridIndex2 back() {
        return _getUnflattenedIndex(_indices.back());
    }

    inline void push_back(GridIndex2 g) {
        _indices.push_back(_getFlatIndex(g));
    }

    inline void push_back(int i, int j) {
        _indices.push_back(_getFlatIndex(i, j));
    }

    void insert(std::vector<GridIndex2> &indices);
    void insert(GridIndexVector &indices);

    inline void pop_back() {
        _indices.pop_back();
    }

    inline void clear() {
        _indices.clear();
    }

    std::vector<GridIndex2> getVector();
    void getVector(std::vector<GridIndex2> &vector);

    int width = 0;
    int height = 0;

private:

    inline unsigned int _getFlatIndex(int i, int j) {
        return (unsigned int)i + (unsigned int)width *
               ((unsigned int)j);
    }

    inline unsigned int _getFlatIndex(GridIndex2 g) {
        return (unsigned int)g.i + (unsigned int)width *
               ((unsigned int)g.j);
    }

    inline GridIndex2 _getUnflattenedIndex(unsigned int flatidx) {
        int i = flatidx % width;
        int j = (flatidx / width);

        return GridIndex2(i, j);
    }

    std::vector<int> _indices;

};

struct PressureSolverParameters {
    double cellwidth;
    double density;
    double deltaTime;

    GridIndexVector *fluidCells;
    MaterialGridData2 *materialGrid;
    MACVelocityGrid2 *velocityField;
};

class VectorXd{
    public:
    VectorXd();
    VectorXd(int size);
    VectorXd(int size, double fill);
    VectorXd(VectorXd &vector);
    ~VectorXd();

    double operator [](int i) const;
    double& operator[](int i);

    inline size_t size() {
        return _vector.size();
    }

    void fill(double fill);
    double dot(VectorXd &vector);
    double absMaxCoeff();

    std::vector<double> _vector;

};

struct MatrixCell {
    char diag;
    char plusi;
    char plusj;
    char plusk;

    MatrixCell() : diag(0x00), plusi(0x00), plusj(0x00), plusk(0x00) {}
};

class GridIndexKeyMap{
public:
    GridIndexKeyMap();
    GridIndexKeyMap(int i, int j);
    ~GridIndexKeyMap();

    void clear();
    void insert(GridIndex2 g, int key);
    void insert(int i, int j, int key);
    int find(GridIndex2 g);
    int find(int i, int j);

private:

    inline unsigned int _getFlatIndex(int i, int j){
        i = Min(i, _isize-1);
        j = Min(j, _isize-1);
        return (unsigned int)i + (unsigned int)_isize *
               ((unsigned int)j);
    }

    inline unsigned int _getFlatIndex(GridIndex2 g){
        unsigned int i = Min(g.i, _isize-1);
        unsigned int j = Min(g.j, _isize-1);
        return (unsigned int)i + (unsigned int)_isize *
               ((unsigned int)j);
    }

    int _isize = 0;
    int _jsize = 0;

    std::vector<int> _indices;
    int _notFoundValue = -1;

};

class MatrixCoefficients{
    public:
    MatrixCoefficients();
    MatrixCoefficients(int size);
    ~MatrixCoefficients();

    const MatrixCell operator [](int i) const;
    MatrixCell& operator [](int i);

    inline size_t size() {
        return cells.size();
    }

    std::vector<MatrixCell> cells;
};

class TmpPressureSolver
{
public:
    TmpPressureSolver();
    ~TmpPressureSolver();

    void solve(PressureSolverParameters params, VectorXd &pressure);

private:

    inline int _GridToVectorIndex(GridIndex2 g) {
        return  _keymap.find(g);
    }
    inline int _GridToVectorIndex(int i, int j) {
        return _keymap.find(i, j);
    }
    inline GridIndex2 _VectorToGridIndex(int i) {
        return _fluidCells->at(i);
    }

    void _initialize(PressureSolverParameters params);
    void _initializeGridIndexKeyMap();
    void _calculateNegativeDivergenceVector(VectorXd &b);
    void _calculateMatrixCoefficients(MatrixCoefficients &A);
    int _getNumFluidOrAirCellNeighbours(int i, int j);
    void _calculatePreconditionerVector(MatrixCoefficients &A, VectorXd &precon);
    void _solvePressureSystem(MatrixCoefficients &A,
                              VectorXd &b,
                              VectorXd &precon,
                              VectorXd &pressure);
    void _applyPreconditioner(MatrixCoefficients &A,
                              VectorXd &precon,
                              VectorXd &residual,
                              VectorXd &vect);
    void _applyMatrix(MatrixCoefficients &A, VectorXd &x, VectorXd &result);
    void _addScaledVector(VectorXd &v1, VectorXd &v2, double scale);
    void _addScaledVectors(VectorXd &v1, double s1,
                           VectorXd &v2, double s2,
                           VectorXd &result);
    bool isCellSolid(int i, int j);
    bool isCellFluid(int i, int j);

    int _isize = 0;
    int _jsize = 0;
    double _dx = 0;
    double _density = 0;
    double _deltaTime = 0;
    int _matSize = 0;

    double _pressureSolveTolerance = 1e-6;
    int _maxCGIterations = 200;

    GridIndexVector *_fluidCells;
    MaterialGridData2 *_materialGrid;
    MACVelocityGrid2 *_vField;
    GridIndexKeyMap _keymap;
};
