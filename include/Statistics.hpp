#include "Common.hpp"
#include "GaussianMixture.hpp"
#include <iostream>
#include <numeric>
#include <random>

namespace gm {

template <int Dim>
std::vector<Vector<Dim>>
draw_from_gaussian_mixture(const GaussianMixture<Dim> &gmm, size_t n_samples) {

  static std::mt19937 gen{std::random_device{}()};
  static std::normal_distribution<> nd;

  std::vector<Vector<Dim>> samples;
  samples.reserve(n_samples);

  std::vector<double> weights;
  weights.reserve(gmm.get_size());
  for (auto it = gmm.cbegin(); it < gmm.cend(); ++it)
    weights.push_back(it->get_weight());
  std::discrete_distribution<> dd(weights.begin(), weights.end());

  for (size_t i = 0; i < n_samples; ++i) {
    const auto &component = gmm[dd(gen)];
    Eigen::SelfAdjointEigenSolver<Matrix<Dim, Dim>> eigen_solver(
        component.get_covariance());
    const auto transform = eigen_solver.eigenvectors() *
                           eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
    const auto sample =
        component.get_mean() +
        transform * Vector<Dim>{}.unaryExpr([](auto x) { return nd(gen); });
    samples.push_back(sample);
  }
  return samples;
}

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
