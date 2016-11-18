#ifndef MODIFIED_SCHWARZ_HPP
#define MODIFIED_SCHWARZ_HPP

#include <complex>
#include <cmath>
#include <armadillo>

namespace ModifiedSchwarz
{

typedef std::complex<double> ComplexDouble;

typedef arma::Mat<ComplexDouble> cmatd;
typedef arma::Col<ComplexDouble> cvecd;
typedef arma::Col<double> vecd;

const ComplexDouble i2pi(0.0, 2.0*arma::datum::pi);

////////////////////////////////////////////////////////////////////////
/*!
 * Represents unit circle domain by holding vectors of center and radii of the
 * inner circles.
 */
class UnitCircleDomain
{
    cvecd _centers;
    vecd _radii;

public:
    UnitCircleDomain(const cvecd &centers, const vecd &radii)
        : _centers(centers), _radii(radii) {}

    const cvecd &getCenters() const { return _centers; }
    const vecd &getRadii() const { return _radii; }

    unsigned connectivity() const { return unsigned(_centers.n_elem) + 1; }
    unsigned m() const { return unsigned(_centers.n_elem); }
};

/*!
 * Example domain defined in matlab code (it's cannon at this point):
 *
 *   D = circleRegion(...
 *              circle(0, 1), ...
 *              circle(-0.2517+0.3129i, 0.2377), ...
 *              circle(0.2307-0.4667i, 0.1557));
 *
 */
UnitCircleDomain domainExample3();

////////////////////////////////////////////////////////////////////////
/*!
 * Container class for the matrix used in the Schwarz problem spectral solver.
 */
class SpectralMatrix
{
    cmatd _theMatrix;

protected:
    void constructMatrix(const UnitCircleDomain& domain, const uint truncation);

public:
    SpectralMatrix(const UnitCircleDomain& domain, const uint truncation)
    { constructMatrix(domain, truncation); }

    const cmatd& getMatrix() const { return _theMatrix; }
};

////////////////////////////////////////////////////////////////////////
/*!
 * Abstract representation of solution to the Schwarz problem.
 */
class SchwarzSolution
{
    cmatd _domainMatrix;

public:
    virtual cvecd eval(const cvecd& points) = 0;
};

////////////////////////////////////////////////////////////////////////
/*!
 * Solution to Schwarz problem using the spectral method.
 */
class SpectralSolution : public SchwarzSolution
{
    cmatd _coefficients;

public:
    SpectralSolution(const cmatd& coefficients)
        : _coefficients(coefficients) {}

    virtual cvecd eval(const cvecd& points) { return cvecd(); }
};

////////////////////////////////////////////////////////////////////////
// Forward declaration.
class SchwarzProblem;

////////////////////////////////////////////////////////////////////////
/*!
 * Abstract representation of method to solve Schwarz problem.
 */
class SchwarzSolver
{
public:
    virtual ~SchwarzSolver() {}

    virtual SchwarzSolution solve(const SchwarzProblem& problem) = 0;
    virtual SchwarzSolution solve(const SchwarzProblem& problem, SchwarzSolution& previous) = 0;
};

////////////////////////////////////////////////////////////////////////
/*!
 * Spectral Schwarz solver.
 */
class SolverSpectral : public SchwarzSolver
{
    SpectralMatrix _domainMatrix;

public:
    SolverSpectral(const SolverSpectral& other) : _domainMatrix(other.getMatrix()) {}
    virtual ~SolverSpectral() {}

    const SpectralMatrix& getMatrix() const { return _domainMatrix; }

    virtual SchwarzSolution solve(const SchwarzProblem& problem)
    { return SpectralSolution(cmatd()); }

    virtual SchwarzSolution solve(const SchwarzProblem& problem,
            SchwarzSolution& previous)
    { return SpectralSolution(cmatd()); }
};

////////////////////////////////////////////////////////////////////////
/*!
 * Abstract representation of the Schwarz problem.
 *
 * Schwarz problem is stored as a matrix of boundary data where each column
 * represents a boundary.
 *
 * Provides a solve() method which creates and returns a SchwarzSolution.
 *
 */
class SchwarzProblem
{
    cmatd _problemData;

protected:
    SchwarzSolver* _solver;

public:
    SchwarzProblem(const cmatd& boundaryData)
        : _problemData(boundaryData), _solver(new SolverSpectral) {}
//    SchwarzProblem(const SchwarzProblem& other)
//        : _problemData(other.data()), _solver(new SolverSpectral) {}

    ~SchwarzProblem() { delete _solver; }

    SchwarzSolution solve() { return _solver->solve(*this); }

    const cmatd& data() const { return _problemData; }
};

}; // namespace ModifiedSchwarz

#endif // MODIFIED_SCHWARZ_HPP