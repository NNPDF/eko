#include "../../../types.hpp"
#include "../../../harmonics/cache.hpp"
#include "./as1.hpp"


namespace ekorepp {
namespace anomalous_dimensions {
namespace unpolarized {
namespace spacelike {

/**
 * @brief Compute the tower of the non-singlet anomalous dimensions.
 * @param order_qcd QCD perturbative order
 * @param mode sector identifier
 * @param c Mellin cache
 * @param nf number of active flavors
 * @return Unpolarized, space-like, non-singlet anomalous dimensions
 */
vctr gamma_ns(const unsigned int order_qcd, const unsigned int mode, harmonics::Cache& c, const unsigned int nf) {
    vctr gamma_ns(order_qcd);
    gamma_ns[0] = as1::gamma_ns(c, nf);
    return gamma_ns;
}

/**
 * @brief Compute the tower of the singlet anomalous dimension matrices.
 * @param order_qcd QCD perturbative order
 * @param c Mellin cache
 * @param nf Number of light flavors
 * @return Unpolarized, space-like, singlet anomalous dimension matrices
 */
tnsr gamma_singlet(const unsigned int order_qcd, harmonics::Cache& c, const unsigned int nf) {
    tnsr gamma_S(order_qcd);
    gamma_S[0] = as1::gamma_singlet(c, nf);
    return gamma_S;
}

} // namespace spacelike
} // namespace unpolarized
} // namespace anomalous_dimensions
} // namespace ekorepp
