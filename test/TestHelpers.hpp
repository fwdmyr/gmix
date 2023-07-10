#ifndef GMSAM_TEST_HELPERS
#define GMSAM_TEST_HELPERS

#include "../include/GaussianMixture.hpp"
#include <gtest/gtest.h>

namespace test {

static constexpr double RANDOM_TOLERANCE = 5E-2;
static constexpr double DETERMINISTIC_TOLERANCE = 1E-10;

[[nodiscard]] inline bool is_near(double lhs, double rhs, double tolerance) {
  return (lhs - tolerance <= rhs && rhs <= lhs + tolerance);
}

template <int Dim,
          typename std::enable_if_t<gmix::is_unambiguous_v<Dim>, int> = false>
[[nodiscard]] bool is_near(const gmix::ColVector<Dim> &lhs,
                           const gmix::ColVector<Dim> &rhs, double tolerance) {
  for (size_t i = 0; i < Dim; i++) {
    if (!is_near(lhs(i), rhs(i), tolerance))
      return false;
  }
  return true;
}

template <int Dim>
[[nodiscard]] bool is_near(const gmix::Matrix<Dim, Dim> &lhs,
                           const gmix::Matrix<Dim, Dim> &rhs,
                           double tolerance) {
  for (size_t i = 0; i < Dim; i++) {
    if (!is_near(lhs(i, i), rhs(i, i), tolerance))
      return false;
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

template <int Dim>
[[nodiscard]] bool is_near(const gmix::GaussianMixture<Dim> &lhs,
                           const gmix::GaussianMixture<Dim> &rhs,
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

template <int Dim>
[[nodiscard]] bool compare_gaussian_mixtures(gmix::GaussianMixture<Dim> &lhs,
                                             gmix::GaussianMixture<Dim> &rhs,
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
