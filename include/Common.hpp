#ifndef GMSAM_COMMON_HPP
#define GMSAM_COMMON_HPP

#include <Eigen/Dense>

namespace gm {

template <int Rows, int Cols> using Matrix = Eigen::Matrix<double, Rows, Cols>;

using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

template <int Rows> using DiagonalMatrix = Eigen::DiagonalMatrix<double, Rows>;

using DiagonalMatrixX = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

template <int Rows> using Vector = Eigen::Matrix<double, Rows, 1>;

using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;

} // namespace gm
#endif // !GMSAM_COMMON_HPP
