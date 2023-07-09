#ifndef GMSAM_COMMON_HPP
#define GMSAM_COMMON_HPP

#include "TypeTraits.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <initializer_list>
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

template <int Rows> using ColVector = Eigen::Matrix<double, Rows, 1>;

using ColVectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;

template <int Cols> using RowVector = Eigen::Matrix<double, 1, Cols>;

using RowVectorX = Eigen::Matrix<double, 1, Eigen::Dynamic>;

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_static_v<MatrixTypeTraits<MatrixType>::Rows> &&
                  gm::is_static_v<MatrixTypeTraits<MatrixType>::Cols>,
              int> = 0>
[[nodiscard]] MatrixType
initialize(std::initializer_list<std::initializer_list<
               typename MatrixTypeTraits<MatrixType>::ElementType>> &&list) {
  assert(list.size() == MatrixTypeTraits<MatrixType>::Rows);
  for (const auto &row : list)
    assert(row.size() == MatrixTypeTraits<MatrixType>::Cols);
  MatrixType A{MatrixTypeTraits<MatrixType>::Rows,
               MatrixTypeTraits<MatrixType>::Cols};
  auto i = 0;
  for (const auto row : list) {
    auto j = 0;
    for (const auto element : row) {
      A(i, j) = element;
      j++;
    }
    i++;
  }
  return A;
}

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_static_v<MatrixTypeTraits<MatrixType>::Rows> &&
                  gm::is_dynamic_v<MatrixTypeTraits<MatrixType>::Cols>,
              int> = 0>
[[nodiscard]] MatrixType
initialize(std::initializer_list<std::initializer_list<
               typename MatrixTypeTraits<MatrixType>::ElementType>> &&list) {
  const auto cols = list.begin()->size();
  assert(list.size() == MatrixTypeTraits<MatrixType>::Rows);
  for (const auto &row : list)
    assert(row.size() == cols);
  MatrixType A{MatrixTypeTraits<MatrixType>::Rows, cols};
  auto i = 0;
  for (const auto row : list) {
    auto j = 0;
    for (const auto element : row) {
      A(i, j) = element;
      j++;
    }
    i++;
  }
  return A;
}

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_dynamic_v<MatrixTypeTraits<MatrixType>::Rows> &&
                  gm::is_static_v<MatrixTypeTraits<MatrixType>::Cols>,
              int> = 0>
[[nodiscard]] MatrixType
initialize(std::initializer_list<std::initializer_list<
               typename MatrixTypeTraits<MatrixType>::ElementType>> &&list) {
  const auto rows = list.size();
  assert(list.size() == rows);
  for (const auto &row : list)
    assert(row.size() == MatrixTypeTraits<MatrixType>::Cols);
  MatrixType A{rows, MatrixTypeTraits<MatrixType>::Cols};
  auto i = 0;
  for (const auto row : list) {
    auto j = 0;
    for (const auto element : row) {
      A(i, j) = element;
      j++;
    }
    i++;
  }
  return A;
}

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_dynamic_v<MatrixTypeTraits<MatrixType>::Rows> &&
                  gm::is_dynamic_v<MatrixTypeTraits<MatrixType>::Cols>,
              int> = 0>
[[nodiscard]] MatrixType
initialize(std::initializer_list<std::initializer_list<
               typename MatrixTypeTraits<MatrixType>::ElementType>> &&list) {
  const auto rows = list.size();
  const auto cols = list.begin()->size();
  assert(list.size() == rows);
  for (const auto &row : list)
    assert(row.size() == cols);
  MatrixType A{rows, cols};
  auto i = 0;
  for (const auto row : list) {
    auto j = 0;
    for (const auto element : row) {
      A(i, j) = element;
      j++;
    }
    i++;
  }
  return A;
}

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_unambiguous_v<MatrixTypeTraits<MatrixType>::Cols> &&
                  gm::is_vector_v<MatrixTypeTraits<MatrixType>::Rows> &&
                  gm::is_static_v<MatrixTypeTraits<MatrixType>::Cols>,
              int> = 0>
[[nodiscard]] MatrixType initialize(
    std::initializer_list<typename MatrixTypeTraits<MatrixType>::ElementType>
        &&list) {
  assert(list.size() == MatrixTypeTraits<MatrixType>::Cols);
  MatrixType v{1, gm::is_static_v<MatrixTypeTraits<MatrixType>::Cols>};
  auto i = 0;
  for (const auto element : list) {
    v(0, i) = element;
    i++;
  }
  return v;
}

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_vector_v<MatrixTypeTraits<MatrixType>::Rows> &&
                  gm::is_dynamic_v<MatrixTypeTraits<MatrixType>::Cols>,
              int> = 0>
[[nodiscard]] MatrixType initialize(
    std::initializer_list<typename MatrixTypeTraits<MatrixType>::ElementType>
        &&list) {
  const auto cols = list.size();
  MatrixType v{1, cols};
  auto i = 0;
  for (const auto element : list) {
    v(0, i) = element;
    i++;
  }
  return v;
}

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_unambiguous_v<MatrixTypeTraits<MatrixType>::Rows> &&
                  gm::is_vector_v<MatrixTypeTraits<MatrixType>::Cols> &&
                  gm::is_static_v<MatrixTypeTraits<MatrixType>::Rows>,
              int> = 0>
[[nodiscard]] MatrixType initialize(
    std::initializer_list<typename MatrixTypeTraits<MatrixType>::ElementType>
        &&list) {
  assert(list.size() == MatrixTypeTraits<MatrixType>::Rows);
  MatrixType v{MatrixTypeTraits<MatrixType>::Rows, 1};
  auto i = 0;
  for (const auto element : list) {
    v(i, 0) = element;
    i++;
  }
  return v;
}

template <typename MatrixType,
          typename std::enable_if_t<
              gm::is_vector_v<MatrixTypeTraits<MatrixType>::Cols> &&
                  gm::is_dynamic_v<MatrixTypeTraits<MatrixType>::Rows>,
              int> = 0>
[[nodiscard]] MatrixType initialize(
    std::initializer_list<typename MatrixTypeTraits<MatrixType>::ElementType>
        &&list) {
  const auto rows = list.size();
  MatrixType v{rows, 1};
  auto i = 0;
  for (const auto element : list) {
    v(i, 0) = element;
    i++;
  }
  return v;
}

} // namespace gm
#endif // !GMSAM_COMMON_HPP
