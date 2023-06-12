#ifndef GMSAM_STATISTICS_HPP
#define GMSAM_STATISTICS_HPP

#include "Common.hpp"
#include <iostream>
#include <numeric>
#include <random>

namespace gm {

template <int Dim>
Vector<Dim> sample_mean(const std::vector<Vector<Dim>> &samples) {
  const size_t n = samples.size();
  const auto mu =
      1.0 / n *
      std::accumulate(samples.begin(), samples.end(),
                      static_cast<Vector<Dim>>(Vector<Dim>::Zero()),
                      [](auto acc, const auto &rhs) { return acc + rhs; });
  return mu;
}

template <int Dim>
Matrix<Dim, Dim> sample_covariance(const std::vector<Vector<Dim>> &samples,
                                   const Vector<Dim> &mean) {
  const size_t n = samples.size();
  const auto covariance =
      1.0 / (n - 1) *
      std::accumulate(samples.begin(), samples.end(),
                      static_cast<Matrix<Dim, Dim>>(Matrix<Dim, Dim>::Zero()),
                      [mu = mean](auto acc, const auto &rhs) {
                        return acc + (rhs - mu) * (rhs - mu).transpose();
                      });
  return covariance;
}

} // namespace gm
#endif // !GMSAM_STATISTICS_HPP
