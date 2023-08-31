#ifndef GMSAM_TYPE_TRAITS_HPP
#define GMSAM_TYPE_TRAITS_HPP

#include <eigen3/Eigen/Dense>

namespace gmix {

template <typename> struct MatrixTraits;

template <typename Type, int RowsAtCompileTime, int ColsAtCompileTime>
struct MatrixTraits<Eigen::Matrix<Type, RowsAtCompileTime, ColsAtCompileTime>> {
  using ElementType = Type;
  static constexpr int Rows = RowsAtCompileTime;
  static constexpr int Cols = ColsAtCompileTime;
};

template <typename MatrixT> struct is_dynamic_matrix : std::false_type {};
template <typename T, int RowsAtCompileTime>
struct is_dynamic_matrix<Eigen::Matrix<T, RowsAtCompileTime, Eigen::Dynamic>>
    : std::true_type {};
template <typename T, int ColsAtCompileTime>
struct is_dynamic_matrix<Eigen::Matrix<T, Eigen::Dynamic, ColsAtCompileTime>>
    : std::true_type {};
template <typename T>
struct is_dynamic_matrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
    : std::true_type {};
template <typename MatrixT>
static constexpr auto is_dynamic_v = is_dynamic_matrix<MatrixT>::value;

// TODO: Restrict true case for Eigen matrices
template <typename MatrixT> struct is_matrix : std::true_type {};
template <typename T, int ColsAtCompileTime>
struct is_matrix<Eigen::Matrix<T, 1, ColsAtCompileTime>> : std::false_type {};
template <typename T, int RowsAtCompileTime>
struct is_matrix<Eigen::Matrix<T, RowsAtCompileTime, 1>> : std::false_type {};
template <typename T>
struct is_matrix<Eigen::Matrix<T, 1, 1>> : std::false_type {};
template <typename MatrixT>
static constexpr auto is_matrix_v = is_matrix<MatrixT>::value;

template <typename MatrixT> struct is_column_vector : std::false_type {};
template <typename T, int RowsAtCompileTime>
struct is_column_vector<Eigen::Matrix<T, RowsAtCompileTime, 1>>
    : std::true_type {};
template <typename MatrixT>
static constexpr auto is_column_vector_v = is_column_vector<MatrixT>::value;

template <typename MatrixT> struct is_row_vector : std::false_type {};
template <typename T, int ColsAtCompileTime>
struct is_row_vector<Eigen::Matrix<T, 1, ColsAtCompileTime>> : std::true_type {
};
template <typename MatrixT>
static constexpr auto is_row_vector_v = is_row_vector<MatrixT>::value;

} // namespace gmix

#endif // !GMSAM_TYPE_TRAITS_HPP
