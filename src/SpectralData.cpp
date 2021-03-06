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

#include "SpectralData.h"

namespace ModifiedSchwarz
{

////////////////////////////////////////////////////////////////////////
SpectralData::SpectralData(const UnitCircleDomain& domain)
    : _truncation(SpectralConstants::kSpectralTruncation()),
      _domain(domain),
      _spectralMatrix(constructMatrix())
{}

SpectralData::SpectralData(const UnitCircleDomain& domain, unsigned truncation)
    : _truncation(truncation),
      _domain(domain),
      _spectralMatrix(constructMatrix())
{}

////////////////////////////////////////////////////////////////////////
cx_mat
SpectralData::constructMatrix()
{
    using namespace arma;

    unsigned m = _domain.m();
    cx_vec dv = _domain.centers();
    colvec qv = _domain.radii();

    // Series truncation level.
    unsigned N = _truncation;
    colvec ktmp = -regspace(1., double(N+1));

    // Number of unknowns.
    unsigned Q = m*(N + 1) + N; // N = (Q - m)/(m + 1);

    // The matrix.
    cx_mat L(2*Q, 2*Q, fill::zeros);

    // Double loop construction.
    for (unsigned p = 0; p <= m; ++p)
    {
        unsigned r0 = (p > 0) ? p*N + (p-1) : 0;
        cx_double dp = (p > 0) ? dv(p-1) : 0.;
        double qp = (p > 0) ? qv(p-1) : 1.;

        for (unsigned j = 0; j <= m; ++j)
        {
            unsigned c0 = (j > 0) ? j*N + (j-1) : 0;
            cx_double dj = (j > 0) ? dv(j-1) : 0.;
            double qj = (j > 0) ? qv(j-1) : 1.;

            if (j == 0)
            {
                if (p == 0)
                {
                    // Block L_{0,0}.
                    L(span(0, N-1), span(Q, Q+N-1)).diag().fill(i2pi);
                }
                else
                {
                    // Block L_{p,0}.
                    if (dp == 0.)
                    {
                        cx_vec tmp(N);
                        tmp(0) = i2pi*qp*qp;
                        for (unsigned i = 1; i < N; ++i)
                        {
                            tmp(i) = qp*tmp(i-1);
                        }
                        L(span(r0+1, r0+N), span(0, N-1)).diag() = std::move(tmp);
                    }
                    else
                    {
                        cx_vec tmp(N);
                        tmp(0) = i2pi*qp*dp;
                        for (unsigned i = 1; i < N; ++i)
                        {
                            tmp(i) = dp*tmp(i-1);
                        }
                        L(r0, span(0, N-1)) = std::move(tmp.st());

                        L(r0+1, 0) = i2pi*qp*qp;
                        for (unsigned n = 2; n <= N; ++n)
                        {
                            L(span(r0+1, r0+n), n-1)
                               = L(span(r0, r0+n-1), n-2)%(n*qp/regspace(1., double(n)));
                        }
                    }
                }
            }
            else
            {
                if (p == 0)
                {
                    // Block L_{0,j}.
                    if (dj == 0.)
                    {
                        cx_vec tmp(N);
                        tmp(0) = -i2pi*qj;
                        for (unsigned i = 1; i < N; ++i)
                        {
                            tmp(i) = qj*tmp(i-1);
                        }
                        L(span(0, N-1), span(Q+c0+1, Q+c0+N)).diag() = std::move(tmp);
                    }
                    else
                    {
                        cx_vec tmp(N);
                        tmp(0) = -i2pi*qj;
                        for (unsigned i = 1; i < N; ++i)
                        {
                            tmp(i) = dj*tmp(i-1);
                        }
                        L(span(0, N-1), Q+c0+1) = tmp;

                        for (unsigned n = 3; n <= N+1; ++n)
                        {
                            L(span(n-2, N-1), Q+c0+n-1)
                                = qj*L(span(n-3, N-2), Q+c0+n-2)
                                    %regspace(double(n-2), double(N-1))/double(n-2);
                        }
                    }
                }
                else if(p == j)
                {
                    // Block L_{p,p}.
                    unsigned r0 = p*N + (p-1);
                    L(span(r0, r0+N), span(r0, r0+N)).diag().fill(-i2pi*qv(p-1));
                }
                else
                {
                    // Block L_{p,j}.
                    cx_vec qtmp(N+1);
                    qtmp(0) = qp;
                    cx_vec dtmp(N+1);
                    cx_double djp = dj - dp;
                    dtmp(0) = 1./djp;
                    for (unsigned i = 1; i < N+1; ++i)
                    {
                        qtmp(i) = qp*qtmp(i-1);
                        dtmp(i) = dtmp(i-1)/djp;
                    }
                    L(span(r0, r0+N), Q+c0+1) = -i2pi*qj*qtmp%dtmp;

                    for (unsigned n = 3; n <= N+1; ++n)
                    {
                        L(span(r0, r0+N), Q+c0+n-1)
                            = (qj/djp)*L(span(r0, r0+N), Q+c0+n-2)%(ktmp - double(n-3))/(n-2);
                    }
                }
            }
        }
    }

    L(span(Q, 2*Q-1), span(0, Q-1)) = conj(L(span(0, Q-1), span(Q, 2*Q-1)));
    L(span(Q, 2*Q-1), span(Q, 2*Q-1)) = conj(L(span(0, Q-1), span(0, Q-1)));

    return L;
}

}; // namespace ModifiedSchwarz
