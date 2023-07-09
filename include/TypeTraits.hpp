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

} // namespace gm

#endif // !GMSAM_TYPE_TRAITS_HPP
