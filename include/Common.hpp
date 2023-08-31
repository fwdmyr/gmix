#ifndef GMSAM_COMMON_HPP
#define GMSAM_COMMON_HPP

#include "TypeTraits.hpp"
#include <eigen3/Eigen/Core>
#include <iostream>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

namespace gmix {

template <int Rows, int Cols> using Matrix = Eigen::Matrix<double, Rows, Cols>;

template <int Rows>
using StaticRowsMatrix = Eigen::Matrix<double, Rows, Eigen::Dynamic>;

template <int Cols>
using StaticColsMatrix = Eigen::Matrix<double, Eigen::Dynamic, Cols>;

using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

template <int Rows> using DiagonalMatrix = Eigen::DiagonalMatrix<double, Rows>;

using DiagonalMatrixX = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

template <int Rows, int Cols>
using PermutationMatrix = Eigen::PermutationMatrix<Rows, Cols>;

template <int Rows>
using StaticRowsPermutationMatrix =
    Eigen::PermutationMatrix<Rows, Eigen::Dynamic>;

template <int Cols>
using StaticColsPermutationMatrix =
    Eigen::PermutationMatrix<Eigen::Dynamic, Cols>;

using PermutationMatrixX =
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

template <int Rows> using ColVector = Eigen::Matrix<double, Rows, 1>;

using ColVectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;

template <int Cols> using RowVector = Eigen::Matrix<double, 1, Cols>;

using RowVectorX = Eigen::Matrix<double, 1, Eigen::Dynamic>;

} // namespace gmix
#endif // !GMSAM_COMMON_HPP
