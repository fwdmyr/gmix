#ifndef GMSAM_TEST_HELPERS
#define GMSAM_TEST_HELPERS

#include "../include/GaussianMixture.hpp"
#include <gtest/gtest.h>

namespace test {

static constexpr double RANDOM_TOLERANCE = 5E-2;
static constexpr double DETERMINISTIC_TOLERANCE = 1E-10;

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

template <int Dim>
bool is_near(const gm::GaussianComponent<Dim> &lhs,
             const gm::GaussianComponent<Dim> &rhs, double tolerance) {
  return is_near(lhs.get_weight(), rhs.get_weight(), tolerance) &&
         is_near(lhs.get_mean(), rhs.get_mean(), tolerance) &&
         is_near(lhs.get_covariance(), rhs.get_covariance(), tolerance);
}

template <typename Strategy, int Dim>
bool is_near(const gm::GaussianMixture<Dim, Strategy> &lhs,
             const gm::GaussianMixture<Dim, Strategy> &rhs, double tolerance) {
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

template <typename Strategy, int Dim>
bool compare_gaussian_mixtures(gm::GaussianMixture<Dim, Strategy> &lhs,
                               gm::GaussianMixture<Dim, Strategy> &rhs,
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
