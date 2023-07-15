#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
#define GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP

#include "Common.hpp"
#include "KMeansStrategy.hpp"
#include "Statistics.hpp"
#include <limits>
#include <random>

namespace gmix {

template <int Dim> struct ExpectationMaximizationParameters;
template <int Dim> class ExpectationMaximizationStrategy;

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

template <int Dim> struct ExpectationMaximizationParameters {
  int n_components{0};
  int n_iterations{0};
  double early_stopping_threshold{0.0};
  bool warm_start{false};
};

template <int Dim>
class ExpectationMaximizationStrategy final : public BaseStrategy<Dim> {
public:
  explicit ExpectationMaximizationStrategy(
      const ExpectationMaximizationParameters<Dim> &parameters) noexcept
      : parameters_(parameters) {}
  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const override;

private:
  virtual void initialize(std::vector<GaussianComponent<Dim>> &,
                          const StaticRowsMatrix<Dim> &) const override;

  ExpectationMaximizationParameters<Dim> parameters_{};
};

template <int Dim>
void ExpectationMaximizationStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {

  const KMeansParameters<Dim> initialization_parameters = {
      parameters_.n_components, 1, 0.0, parameters_.warm_start};
  const auto initialization_strategy =
      KMeansStrategy<Dim>{initialization_parameters};
  initialization_strategy.fit(components, samples);
}

template <int Dim>
void ExpectationMaximizationStrategy<Dim>::fit(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {
  if (!parameters_.warm_start || components.size() != parameters_.n_components)
    initialize(components, samples);
  auto responsibilities =
      static_cast<MatrixX>(MatrixX::Zero(components.size(), samples.cols()));
  auto current_log_likelihood = -1.0 * std::numeric_limits<double>::max();
  for (size_t i = 0; i < parameters_.n_iterations; ++i) {
    const auto new_log_likelihood = internal::evaluate_responsibilities(
        components, samples, responsibilities);
    internal::estimate_parameters(samples, responsibilities, components);
    if (new_log_likelihood - current_log_likelihood <
        parameters_.early_stopping_threshold)
      return;
    else
      current_log_likelihood = new_log_likelihood;
  }
}

} // namespace gmix

#endif // !#ifndef GMSAM_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
