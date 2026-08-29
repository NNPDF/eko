#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C-compatible double-precision complex number.
 */
typedef struct {
    double re;
    double im;
} ome_complex_t;

// Space-like unpolarized OMEs at O(as^3) in Mellin N-space
ome_complex_t ome_as3_Agg(ome_complex_t n, unsigned int nf, double L);
ome_complex_t ome_as3_Agq(ome_complex_t n, unsigned int nf, double L);
ome_complex_t ome_as3_Aqg(ome_complex_t n, unsigned int nf, double L);
ome_complex_t ome_as3_AHg(ome_complex_t n, unsigned int nf, double L);
ome_complex_t ome_as3_AHq(ome_complex_t n, unsigned int nf, double L);
ome_complex_t ome_as3_AqqPS(ome_complex_t n, unsigned int nf, double L);
ome_complex_t ome_as3_AqqNS(ome_complex_t n, unsigned int nf, double L, int eta);

#ifdef __cplusplus
}
#endif
