/*
 *  Example of ModifiedSchwarz use.
 *  Here we also compute the derivative.
 */

#include <iostream>
#include <iomanip>
#include <stdlib.h>

#include "Problem.h"

using namespace ModifiedSchwarz;

////////////////////////////////////////////////////////////////////////
int main()
{
    cx_vec dc(2);
    colvec qc(2);
    
    dc(0) = cx_double(-0.5,0.0);
    dc(1) = cx_double(0.5,0.0);
    qc(0) = 0.1;
    qc(1) = 0.1;
    
    UnitCircleDomain D = UnitCircleDomain(dc, qc);
    
    static constexpr unsigned nbp = 256;
    cx_mat zetab = D.boundaryPoints(nbp);
    
    // Create a function analytic in the disk
    // and evaluate it at the boundary points
    cx_double alpha = cx_double(-1.5,0.0);
    cx_mat f = log(zetab-alpha);
    
    // Set up and solve MS problem
    RealBoundaryValues RHS(BoundaryPoints(D,nbp),imag(f));
    Problem MS(RHS);
    Solution sol = MS.solve();
    
    cx_mat sv = reshape(sol(vectorise(zetab)), zetab.n_rows, zetab.n_cols);
    double ic = real(log(zetab(0,0)-alpha)-sv(0,0)); // Normalisation constant
    
    // Also get the 2nd derivative
    int repd = 2;
    Solution dsol = sol.diff(repd);
    
    // Eval at z
    cx_vec z{cx_double(0.0,0.5), cx_double(0.1,-0.7)};
    cx_vec fn = sol(z)+ic;
    cx_vec dfn = dsol(z);
    
    // Check our analytic and numerical solutions match!
    std::cout << "---------- Function evaluation ---------" << std::endl;
    std::cout << "Analytic" << std::endl;
    std::cout << log(z-alpha) << std::endl;
    std::cout << "Numerical" << std::endl;
    std::cout << fn << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "---------- Derivative evaluation -------" << std::endl;
    std::cout << "Analytic" << std::endl;
    std::cout << -1./pow((z-alpha),2) << std::endl;
    std::cout << "Numerical" << std::endl;
    std::cout << dfn << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}
