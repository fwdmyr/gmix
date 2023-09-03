#ifndef GMSAM_STATISTICS_HPP
#define GMSAM_STATISTICS_HPP

#include "common.hpp"
#include <iostream>
#include <numeric>
#include <random>

namespace gmix {

template <int Dim, template <int> typename FittingStrategy>
class GaussianMixture;

template <template <int> typename FittingStrategy, int Dim>
[[nodiscard]] StaticRowsMatrix<Dim>
draw_from_gaussian_mixture(const GaussianMixture<Dim, FittingStrategy> &gmm,
                           size_t n_samples) {
  static std::mt19937 gen{std::random_device{}()};
  static std::normal_distribution<> nd;

  std::vector<double> weights;
  weights.reserve(gmm.get_size());
  for (auto it = gmm.cbegin(); it < gmm.cend(); ++it)
    weights.push_back(it->get_weight());
  std::discrete_distribution<> dd(weights.begin(), weights.end());
  auto samples = StaticRowsMatrix<Dim>::Zero(Dim, n_samples).eval();

  for (size_t i = 0; i < n_samples; ++i) {
    const auto &component = gmm.get_component(dd(gen));
    Eigen::SelfAdjointEigenSolver<Matrix<Dim, Dim>> eigen_solver(
        component.get_covariance());
    const auto transform = eigen_solver.eigenvectors() *
                           eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
    samples.col(i) =
        component.get_mean() +
        transform * ColVector<Dim>{}.unaryExpr([](auto x) { return nd(gen); });
  }
  return samples;
}

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
