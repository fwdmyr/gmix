#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_IMPL_HPP
#define GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_IMPL_HPP

#include "kmeans_strategy.hpp"

namespace gmix {

namespace internal {

template <int Dim>
[[nodiscard]] double
evaluate_responsibilities(const std::vector<GaussianComponent<Dim>> &components,
                          const StaticRowsMatrix<Dim> &samples,
                          MatrixX &responsibilities) {
  const auto n_samples = samples.cols();
  const auto n_components = components.size();
  auto log_likelihood = 0.0;
  for (size_t i = 0; i < n_samples; ++i) {
    const auto sample = static_cast<ColVector<Dim>>(samples.col(i));
    auto responsibility =
        static_cast<ColVectorX>(ColVectorX::Zero(n_components, 1));
    for (size_t j = 0; j < n_components; ++j) {
      const auto &component = components[j];
      responsibility(j) = component(sample);
    }
    const auto responsibility_sum = responsibility.sum();
    log_likelihood += std::log(responsibility_sum);
    responsibilities.col(i) = 1.0 / responsibility_sum * responsibility;
  }
  return log_likelihood;
}

template <int Dim>
void estimate_parameters(const StaticRowsMatrix<Dim> &samples,
                         const MatrixX &responsibilities,
                         std::vector<GaussianComponent<Dim>> &components) {
  const auto n_samples = samples.cols();
  const auto n_samples_responsible =
      static_cast<ColVectorX>(responsibilities.rowwise().sum());
  for (size_t i = 0; i < components.size(); ++i) {
    auto &component = components[i];
    component.set_weight(1.0 / n_samples * n_samples_responsible(i));
    const auto mu = 1.0 / n_samples_responsible(i) *
                    static_cast<ColVector<Dim>>(
                        (samples.transpose().array().colwise() *
                         responsibilities.row(i).transpose().array())
                            .transpose()
                            .rowwise()
                            .sum());
    component.set_mean(mu);
    const auto centered_samples = static_cast<StaticRowsMatrix<Dim>>(
        samples.colwise() - component.get_mean());
    const auto weighted_centered_samples = static_cast<StaticRowsMatrix<Dim>>(
        (centered_samples.transpose().array().colwise() *
         responsibilities.row(i).transpose().array())
            .transpose());
    // TODO: Can Eigen's reductions handle the sum over the outer products?
    // Maybe user-defined redux?
    auto sigma = static_cast<Matrix<Dim, Dim>>(Matrix<Dim, Dim>::Zero());
    for (size_t j = 0; j < n_samples; ++j) {
      sigma += weighted_centered_samples.col(j) *
               centered_samples.transpose().row(j);
    }
    sigma *= 1.0 / n_samples_responsible(i);
    component.set_covariance(sigma);
  }
}

} // namespace internal

} // namespace gmix

#endif // !#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_IMPL_HPP
