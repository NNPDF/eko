#include <vector>
#include "./as1.hpp"
#include "../../../harmonics/polygamma.hpp"


namespace ekorepp {
namespace anomalous_dimensions {
namespace unpolarized {
namespace spacelike {

/**
 * @brief Compute the tower of the non-singlet anomalous dimensions.
 * @param order_qcd QCD perturbative orders
 * @param mode sector identifier
 * @param n Mellin variable
 * @param nf number of active flavors
 * @return non-singlet anomalous dimensions
 */
std::vector<std::complex<double>> gamma_ns(const unsigned int order_qcd, const unsigned int mode, std::complex<double> n, const unsigned int nf) {
    std::vector<std::complex<double>> gamma_ns(order_qcd);
    gamma_ns[0] = as1::gamma_ns(n, nf);
    return gamma_ns;
}

} // namespace spacelike
} // namespace unpolarized
} // namespace anomalous_dimensions
} // namespace ekorepp
