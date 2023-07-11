#ifndef EKOREPP_HARMONICS_W1_HPP_
#define EKOREPP_HARMONICS_W1_HPP_

#include "../types.hpp"
#include "./polygamma.hpp"

namespace ekorepp {
namespace harmonics {
namespace w1 {

/**
 * @brief Compute the harmonic sum :math:`S_1(N)`.
 * .. math::
 *       S_1(N) = \sum\limits_{j=1}^N \frac 1 j = \psi_0(N+1)+\gamma_E
 *
 * with :math:`\psi_0(N)` the digamma function and :math:`\gamma_E` the
 * Euler-Mascheroni constant.
 * @param N Mellin moment
 * @return (simple) Harmonic sum :math:`S_1(N)`
 */
cmplx S1(const cmplx N) {
    return cern_polygamma(N + 1.0, 0) + 0.5772156649015328606065120;
}

} // namespace w1
} // namespace harmonics
} // namespace ekorepp

#endif // EKOREPP_HARMONICS_W1_HPP_
