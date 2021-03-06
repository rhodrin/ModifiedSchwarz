/*
 * Copyright 2017 Everett Kropf.
 *
 * This file is part of ModifiedSchwarz.
 *
 * ModifiedSchwarz is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ModifiedSchwarz is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ModifiedSchwarz.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Solution.h"

namespace ModifiedSchwarz
{

//////////////////////////////////////////////////////////////////////////////
Solution::Solution(RealInterpolant realPart, colvec constants, RealInterpolant imagPart)
    : _constants(constants)
{
    imagPart.adjustConstants(_constants);
    SDEBUG("create boundary interpolant");
    ComplexInterpolant&& boundary = ComplexInterpolant(realPart, imagPart);

    mat realValues;
    if (realPart.boundaryValues().isEmpty())
    {
        realPart.generateBoundaryValues(imagPart.boundaryValues().points());
    }

    SDEBUG("create boundary values, real size=" << realPart.boundaryValues().values().n_rows
            << "-by-" << realPart.boundaryValues().values().n_cols << " and imag size="
            << imagPart.boundaryValues().values().n_rows << "-by-"
            << imagPart.boundaryValues().values().n_cols);
    ComplexBoundaryValues&& values = ComplexBoundaryValues(
            realPart.boundaryValues().points(),
            cx_mat(realPart.boundaryValues().values(), imagPart.boundaryValues().values())
            );
    SDEBUG("create closure interpolant");
    _interpolant = ClosureInterpolant(boundary, CauchyInterpolant(values));
}

Solution::Solution(RealInterpolant realPart, colvec constants, RealInterpolant imagPart, SolverData::Ptr pSolverData)
{
    *this = Solution(realPart, constants, imagPart);
    _pSolverData = pSolverData;
}

}; // namespace ModifiedSchwarz
