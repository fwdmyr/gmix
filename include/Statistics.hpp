#ifndef GMSAM_STATISTICS_HPP
#define GMSAM_STATISTICS_HPP

#include "Common.hpp"
#include <iostream>
#include <numeric>
#include <random>

namespace gm {

namespace internal {

template <int Dim>
Vector<Dim> sample_mean(const StaticRowsMatrix<Dim> &samples) {
  return samples.rowwise().mean();
}

template <int Dim>
Matrix<Dim, Dim> sample_covariance(const StaticRowsMatrix<Dim> &samples) {
  const auto mu = sample_mean(samples);
  const auto centered_samples = samples.colwise() - mu;
  return (centered_samples * centered_samples.transpose()) /
         static_cast<double>(samples.cols() - 1);
}

template <int Dim>
Matrix<Dim, Dim> sample_covariance(const StaticRowsMatrix<Dim> &samples,
                                   const Vector<Dim> &mu) {
  const auto centered_samples = samples.colwise() - mu;
  return (centered_samples * centered_samples.transpose()) /
         static_cast<double>(samples.cols() - 1);
}

} // namespace internal

} // namespace gm

#endif // !GMSAM_STATISTICS_HPP
