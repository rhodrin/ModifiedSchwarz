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

Solution Solution::rdftDerivative(Solution sol, int n){
    
    while (n >= 1){
        sol = dftDerivative(sol);
        n=n-1;
    }
    
    return sol;
}

Solution Solution::dftDerivative(Solution sol){
    
    int nf = 256;
    
    auto D = _interpolant.boundary().domain();
    cx_vec d = D.centers();
    colvec q = D.radii();
    int M = D.connectivity();
    
    cx_mat zf = D.boundaryPoints(nf);
    cx_mat sv = reshape(sol(vectorise(zf)), zf.n_rows, zf.n_cols);
    
    mat p1 = arma::regspace(0,nf/2-1);
    mat p2(1,1,arma::fill::zeros);
    mat p3 = arma::regspace(-nf/2+1,-1);
    mat dmult = join_cols(join_cols(p1, p2), p3);
    
    cx_mat eij(nf,M,arma::fill::zeros);
    for(int i=0;i<M;++i){
        if(i==0){
            eij.col(i) = zf.col(i);
        }else{
            eij.col(i) = (zf.col(i)-d(i-1))/q(i-1);
        }
    }

    cx_mat dk(nf,M,arma::fill::zeros);
    cx_mat p(nf,M);
    for(int j=0;j<M;++j){
        dk.col(j) = cx_double(0.0,1.0)*dmult%fft(sv.col(j))/nf;
        p.col(j) = polyval(flipud(dk.rows(0,nf/2-1).col(j)), eij.col(j)) 
                   +polyval(dk.rows(nf/2,nf-1).col(j), 1./eij.col(j));
        
    }
    
    cx_mat val(nf,M);
    for(int i=0;i<M;++i){
        if(i==0){
            val.col(i) = p.col(i)/(cx_double(0.0,1.0)*eij.col(i));
        }else{
            val.col(i) = p.col(i)/(cx_double(0.0,q(i-1))*eij.col(i));
        }
    }
    
    // Now build a new solution
    RealBoundaryValues dsr(BoundaryPoints(D,nf),real(val));
    RealBoundaryValues dsi(BoundaryPoints(D,nf),imag(val));
    colvec constants = sol.constants();
    
    return Solution(dsr, imag(constants), dsi);
}

}; // namespace ModifiedSchwarz
