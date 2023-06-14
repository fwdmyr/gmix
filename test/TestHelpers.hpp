#ifndef GMSAM_TEST_HELPERS
#define GMSAM_TEST_HELPERS

#include "../include/GaussianMixture.hpp"
#include <gtest/gtest.h>

namespace test {

static constexpr double THRESHOLD = 5E-2;

inline bool is_near(double lhs, double rhs, double tolerance) {
  return (lhs - tolerance <= rhs && rhs <= lhs + tolerance);
}

template <int Dim> static constexpr bool is_unambiguous_v = Dim > 1;

template <int Dim, typename std::enable_if_t<is_unambiguous_v<Dim>, int> = 0>
bool is_near(const gm::Vector<Dim> &lhs, const gm::Vector<Dim> &rhs,
             double tolerance) {
  for (size_t i = 0; i < Dim; i++) {
    if (!is_near(lhs(i), rhs(i), tolerance))
      return false;
  }
  return true;
}

template <int Dim>
bool is_near(const gm::Matrix<Dim, Dim> &lhs, const gm::Matrix<Dim, Dim> &rhs,
             double tolerance) {
  for (size_t i = 0; i < Dim; i++) {
    if (!is_near(lhs(i, i), rhs(i, i), tolerance))
      return false;
  }
  return true;
}

template <typename Strategy, int Dim>
bool is_near(const gm::GaussianMixture<Strategy, Dim> &lhs,
             const gm::GaussianMixture<Strategy, Dim> &rhs, double tolerance) {
  if (lhs.get_size() != rhs.get_size())
    return false;
  for (int i = 0; i < lhs.get_size(); ++i) {
    const auto &lhs_component = lhs.get_component(i);
    const auto &rhs_component = rhs.get_component(i);
    if (!is_near(lhs_component.get_weight(), rhs_component.get_weight(),
                 tolerance) ||
        !is_near(lhs_component.get_mean(), rhs_component.get_mean(),
                 tolerance) ||
        !is_near(lhs_component.get_covariance(), rhs_component.get_covariance(),
                 tolerance))
      return false;
  }
  return true;
}

template <typename Strategy, int Dim>
bool compare_gaussian_mixtures(gm::GaussianMixture<Strategy, Dim> &lhs,
                               gm::GaussianMixture<Strategy, Dim> &rhs,
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
