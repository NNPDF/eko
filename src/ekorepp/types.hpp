#ifndef EKOREPP_TYPES_HPP_
#define EKOREPP_TYPES_HPP_

#include <complex>
#include <vector>

namespace ekorepp {

typedef std::complex<double> cmplx;
typedef std::vector<cmplx> vctr;
typedef std::vector<vctr> mtrx;
typedef std::vector<mtrx> tnsr;

} // namespace ekorepp

#endif // EKOREPP_TYPES_HPP_
