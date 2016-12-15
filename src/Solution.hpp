#ifndef SOLUTION_HPP
#define SOLUTION_HPP

#include <memory>

#include "SchwarzTypes.hpp"
#include "RealInterpolant.hpp"

namespace ModifiedSchwarz
{

class SolverData;
using SolverDataSPtr = std::shared_ptr<SolverData>;

///////////////////////////////////////////////////////////////////////////
/*!
 * Solution is made of 3 parts:
 *   1. an imaginary part (RealInterpolant) given as part of the problem,
 *   2. a real part (RealInterpolant) which is found by the solver, and
 *   3. an imaginary constant for each boundary also found by the solver.
 * All data is on the boundary of the domain.
 *
 * In addition, a SolverData object may be stored to accelerate future
 * solver runs in the same domain.
 */
class Solution
{
    RealInterpolant _realPart;
    colvec _constants;
    RealInterpolant _imagPart;
    SolverDataSPtr _solverDataPtr;

public:
    Solution(RealInterpolant realPart, colvec constants, RealInterpolant imagPart)
        : _realPart(realPart), _constants(constants), _imagPart(imagPart) {};

    Solution(RealInterpolant realPart, colvec constants, RealInterpolant imagPart,
             SolverDataSPtr solverDataPtr)
        : _realPart(realPart), _constants(constants), _imagPart(imagPart),
          _solverDataPtr(solverDataPtr) {};

    const RealInterpolant& realPart() const { return _realPart; }
    const colvec& constants() const { return _constants; }
    const RealInterpolant& imagPart() const { return _imagPart; }
    SolverDataSPtr solverDataPtr() { return _solverDataPtr; }

    cx_vec eval(const cx_vec&);
};

}; // namespace ModifiedSchwarz

#endif // SOLUTION_HPP