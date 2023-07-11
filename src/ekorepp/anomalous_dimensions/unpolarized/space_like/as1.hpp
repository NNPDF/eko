#include "../../../constants.hpp"
#include "../../../types.hpp"
#include "../../../harmonics/cache.hpp"


namespace ekorepp {
namespace anomalous_dimensions {
namespace unpolarized {
namespace spacelike {
namespace as1 {

/**
 * @brief Compute the leading-order non-singlet anomalous dimension.
 * Implements Eq. (3.4) of :cite:`Moch:2004pa`.
 * @param c Mellin cache
 * @param _nf Number of light flavors
 * @return Leading-order non-singlet anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
 */
cmplx gamma_ns(harmonics::Cache& c, const unsigned int _nf) {
    const cmplx N = c.N();
    const cmplx S1 = c.get(harmonics::K::S1);
    const cmplx gamma = -(3.0 - 4.0 * S1 + 2.0 / N / (N + 1.0));
    return CF * gamma;
}

/**
 * @brief Compute the leading-order quark-gluon anomalous dimension.
 * Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
 * @param c Mellin cache
 * @param nf Number of light flavors
 * @return Leading-order quark-gluon anomalous dimension :math:`\\gamma_{qg}^{(0)}(N)`
 */
cmplx gamma_qg(harmonics::Cache& c, const unsigned int nf) {
    const cmplx N = c.N();
    const cmplx gamma = -(pow(N, 2) + N + 2.0) / (N * (N + 1.0) * (N + 2.0));
    return 2.0 * TR * 2.0 * nf * gamma;
}

/**
 * @brief Compute the leading-order gluon-quark anomalous dimension.
 * Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
 * @param c Mellin cache
 * @param _nf Number of light flavors
 * @return Leading-order gluon-quark anomalous dimension :math:`\\gamma_{gq}^{(0)}(N)`
 */
cmplx gamma_gq(harmonics::Cache& c, const unsigned int _nf) {
    const cmplx N = c.N();
    const cmplx gamma = -(pow(N, 2) + N + 2.0) / (N * (N + 1.0) * (N - 1.0));
    return 2.0 * CF * gamma;
}

/**
 * @brief Compute the leading-order gluon-gluon anomalous dimension.
 * Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
 * @param c Mellin cache
 * @param nf Number of light flavors
 * @return Leading-order gluon-gluon anomalous dimension :math:`\\gamma_{gg}^{(0)}(N)`
 */
cmplx gamma_gg(harmonics::Cache& c, const unsigned int nf) {
    const cmplx N = c.N();
    const cmplx S1 = c.get(harmonics::K::S1);
    const cmplx gamma = S1 - 1.0 / N / (N - 1.0) - 1.0 / (N + 1.0) / (N + 2.0);
    return CA * (4.0 * gamma - 11.0 / 3.0) + 4.0 / 3.0 * TR * nf;
}

/**
 * @brief Compute the leading-order singlet anomalous dimension matrix.
 * @param c Mellin cache
 * @param nf Number of light flavors
 * @return Leading-order singlet anomalous dimension matrix :math:`\\gamma_{S}^{(0)}(N)`
 */
mtrx gamma_singlet(harmonics::Cache& c, const unsigned int nf) {
    const cmplx gamma_qq = gamma_ns(c, nf);
    return {{gamma_qq, gamma_qg(c, nf)}, {gamma_gq(c, nf), gamma_gg(c, nf)}};
}

} // namepsace as1
} // namespace spacelike
} // namespace unpolarized
} // namespace anomalous_dimensions
} // namespace ekorepp
