#ifndef GMSAM_COMMON_HPP
#define GMSAM_COMMON_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

namespace gm {

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

template <int Rows> using Vector = Eigen::Matrix<double, Rows, 1>;

using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;

} // namespace gm
#endif // !GMSAM_COMMON_HPP
