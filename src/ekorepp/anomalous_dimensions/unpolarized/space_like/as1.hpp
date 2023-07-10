#include "../../../constants.hpp"
#include "../../../harmonics/w1.hpp"


namespace ekorepp {
namespace anomalous_dimensions {
namespace unpolarized {
namespace spacelike {
namespace as1 {

std::complex<double> gamma_ns(std::complex<double> N, const unsigned int _nf) {
    const std::complex<double> S1 = harmonics::S1(N);
    const std::complex<double> gamma = -(3.0 - 4.0 * S1 + 2.0 / N / (N + 1.0));
    return CF * gamma;
}

} // namepsace as1
} // namespace spacelike
} // namespace unpolarized
} // namespace anomalous_dimensions
} // namespace ekorepp
