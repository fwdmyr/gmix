#ifndef GMSAM_COMMON_HPP
#define GMSAM_COMMON_HPP

#include "gmix_traits.hpp"
#include <eigen3/Eigen/Core>

namespace gmix {

template <int Rows, int Cols> using Matrix = Eigen::Matrix<double, Rows, Cols>;

template <int Rows>
using StaticRowsMatrix = Eigen::Matrix<double, Rows, Eigen::Dynamic>;

template <int Cols>
using StaticColsMatrix = Eigen::Matrix<double, Eigen::Dynamic, Cols>;

using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

template <int Rows> using ColVector = Eigen::Matrix<double, Rows, 1>;

using ColVectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;

template <int Cols> using RowVector = Eigen::Matrix<double, 1, Cols>;

using RowVectorX = Eigen::Matrix<double, 1, Eigen::Dynamic>;

} // namespace gmix
#endif // !GMSAM_COMMON_HPP
