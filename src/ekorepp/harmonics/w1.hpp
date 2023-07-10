#include <math.h>
#include "./polygamma.hpp"

namespace ekorepp {
namespace harmonics {

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
std::complex<double> S1(const std::complex<double> N) {
    return cern_polygamma(N + 1.0, 0) + 0.5772156649015328606065120;
}

} // namespace harmonics
} // namespace ekorepp
