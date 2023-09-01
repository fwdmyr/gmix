#ifndef GMSAM_TEST_HELPERS
#define GMSAM_TEST_HELPERS

#include "../include/gaussian_mixture.hpp"
#include <gtest/gtest.h>
#include <iostream>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

namespace test {

static constexpr double RANDOM_TOLERANCE = 5E-2;
static constexpr double DETERMINISTIC_TOLERANCE = 1E-10;

[[nodiscard]] inline bool is_near(double lhs, double rhs, double tolerance) {
  return ((lhs - tolerance <= rhs) && (rhs <= lhs + tolerance));
}

template <template <typename, int, int> typename Matrix, typename T,
          int RowsLhs, int ColsLhs, int RowsRhs, int ColsRhs>
[[nodiscard]] bool is_near(const Matrix<T, RowsLhs, ColsLhs> &lhs,
                           const Matrix<T, RowsRhs, ColsRhs> &rhs,
                           double tolerance) {
  if ((lhs.rows() != rhs.rows()) || (lhs.cols() != rhs.cols())) {
    return false;
  }
  for (size_t i = 0; i < lhs.rows(); i++) {
    for (size_t j = 0; j < lhs.cols(); j++) {
      if (!is_near(lhs(i, j), rhs(i, j), tolerance))
        return false;
    }
  }
  return true;
}

template <int Dim>
[[nodiscard]] bool is_near(const gmix::GaussianComponent<Dim> &lhs,
                           const gmix::GaussianComponent<Dim> &rhs,
                           double tolerance) {
  return is_near(lhs.get_weight(), rhs.get_weight(), tolerance) &&
         is_near(lhs.get_mean(), rhs.get_mean(), tolerance) &&
         is_near(lhs.get_covariance(), rhs.get_covariance(), tolerance);
}

template <template <int> typename FittingStrategyLhs,
          template <int> typename FittingStrategyRhs, int Dim>
[[nodiscard]] bool
is_near(const gmix::GaussianMixture<Dim, FittingStrategyLhs> &lhs,
        const gmix::GaussianMixture<Dim, FittingStrategyRhs> &rhs,
        double tolerance) {
  if (lhs.get_size() != rhs.get_size())
    return false;
  for (int i = 0; i < lhs.get_size(); ++i) {
    const auto &lhs_component = lhs.get_component(i);
    const auto &rhs_component = rhs.get_component(i);
    if (!is_near(lhs_component, rhs_component, tolerance))
      return false;
  }
  return true;
}

template <template <int> typename FittingStrategyLhs,
          template <int> typename FittingStrategyRhs, int Dim>
[[nodiscard]] bool
compare_gaussian_mixtures(gmix::GaussianMixture<Dim, FittingStrategyLhs> &lhs,
                          gmix::GaussianMixture<Dim, FittingStrategyRhs> &rhs,
                          double tolerance) {
  if (lhs.get_size() != rhs.get_size())
    return false;

  std::sort(rhs.begin(), rhs.end(), [](const auto &a, const auto &b) {
    return a.get_weight() < b.get_weight();
  });

  do {
    if (is_near(lhs, rhs, tolerance))
      return true;
  } while (std::next_permutation(rhs.begin(), rhs.end(),
                                 [](const auto &a, const auto &b) {
                                   return a.get_weight() < b.get_weight();
                                 }));
  return false;
}

} // namespace test

#endif // !GMSAM_TEST_HELPERS
