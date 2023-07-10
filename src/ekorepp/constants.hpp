namespace ekorepp {

/**
 * @brief The number of colors.
 * Defaults to :math:`N_C = 3`.
 */
const unsigned int NC = 3;

/**
 * @brief The normalization of fundamental generators.
 * Defaults to :math:`T_R = 1/2`.
 */
const double TR = 1.0 / 2.0;

/**
 * @brief Second Casimir constant in the adjoint representation.
 * Defaults to :math:`N_C = 3`.
 */
const double CA = double(NC);

/**
 * @brief Second Casimir constant in the fundamental representation.
 * Defaults to :math:`\frac{N_C^2-1}{2N_C} = 4/3`.
 */
const double CF = double(double(NC * NC - 1) / (2.0 * NC));

}
