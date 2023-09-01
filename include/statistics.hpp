#ifndef GMSAM_STATISTICS_HPP
#define GMSAM_STATISTICS_HPP

#include "common.hpp"
#include <iostream>
#include <numeric>
#include <random>

namespace gmix {

namespace internal {

template <int Dim>
[[nodiscard]] ColVector<Dim> sample_mean(const StaticRowsMatrix<Dim> &samples) {
  return samples.rowwise().mean();
}

template <int Dim>
[[nodiscard]] Matrix<Dim, Dim>
sample_covariance(const StaticRowsMatrix<Dim> &samples) {
  assert(samples.cols() > 1);
  const auto mu = sample_mean(samples);
  const auto centered_samples = samples.colwise() - mu;
  return (centered_samples * centered_samples.transpose()) /
         static_cast<double>(samples.cols() - 1);
}

template <int Dim>
[[nodiscard]] Matrix<Dim, Dim>
sample_covariance(const StaticRowsMatrix<Dim> &samples,
                  const ColVector<Dim> &mu) {
  assert(samples.cols() > 1);
  const auto centered_samples = samples.colwise() - mu;
  return (centered_samples * centered_samples.transpose()) /
         static_cast<double>(samples.cols() - 1);
}

} // namespace internal

} // namespace gmix

#endif // !GMSAM_STATISTICS_HPP
