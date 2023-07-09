#ifndef GMSAM_TYPE_TRAITS_HPP
#define GMSAM_TYPE_TRAITS_HPP

#include <Eigen/Dense>

namespace gm {

template <typename> struct MatrixTypeTraits;

template <typename Type, int RowsAtCompileTime, int ColsAtCompileTime>
struct MatrixTypeTraits<
    Eigen::Matrix<Type, RowsAtCompileTime, ColsAtCompileTime>> {
  using ElementType = Type;
  static constexpr int Rows = RowsAtCompileTime;
  static constexpr int Cols = ColsAtCompileTime;
};

template <int Dim> static constexpr bool is_vector_v = Dim == 1;
template <int Dim> static constexpr bool is_unambiguous_v = Dim > 1;
template <int Dim> static constexpr bool is_dynamic_v = Dim == -1;
template <int Dim> static constexpr bool is_static_v = Dim > 0;
template <int Rows, int Cols>
static constexpr bool is_static_rows_static_cols_v =
    is_static_v<Rows> && is_static_v<Cols>;
template <int Rows, int Cols>
static constexpr bool is_static_rows_dynamic_cols_v =
    is_static_v<Rows> && is_dynamic_v<Cols>;
template <int Rows, int Cols>
static constexpr bool is_dynamic_rows_static_cols_v =
    is_dynamic_v<Rows> && is_static_v<Cols>;
template <int Rows, int Cols>
static constexpr bool is_dynamic_rows_dynamic_cols_v =
    is_dynamic_v<Rows> && is_dynamic_v<Cols>;

} // namespace gm

#endif // !GMSAM_TYPE_TRAITS_HPP
