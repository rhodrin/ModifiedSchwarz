#ifndef SPECTRALDATA_HPP
#define SPECTRALDATA_HPP

#include <memory>

#include "SchwarzTypes.hpp"
#include "UnitCircleDomain.hpp"
#include "SolverData.hpp"
#include "SpectralConstants.hpp"

namespace ModifiedSchwarz
{

////////////////////////////////////////////////////////////////////////
/*!
 * Each domain has a unique spectral matrix (modulo the chosen
 * truncation level for the boundary function series). For each domain
 * then this class should be passed around as a shared pointer. See
 * associated typedef.
 *
 * See [elsewhere] for a description of the matrix.
 */
class SpectralData : public SolverData
{
    unsigned _truncation;
    UnitCircleDomain _domain;
    cx_mat _spectralMatrix;

protected:
    cx_mat constructMatrix();

public:
    SpectralData(const UnitCircleDomain&);
    SpectralData(const UnitCircleDomain&, unsigned);

    using Ptr = std::shared_ptr<SpectralData>;

    const UnitCircleDomain& domain() const { return _domain; }
    const cx_mat& matrix() const { return _spectralMatrix; }
    unsigned truncation() const
    {
        return (matrix().n_cols/2 - _domain.m())/(_domain.m() + 1);
    }

    friend bool operator==(const SpectralData& left, const SpectralData& right)
    {
        return left._domain == right._domain;
    };
};

}; // namespace ModifiedSchwarz

#endif // SPECTRALDATA_HPP
