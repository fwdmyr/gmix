#ifndef GMIX_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
#define GMIX_EXPECTATION_MAXIMIZATION_STRATEGY_HPP

#include "expectation_maximization_policy_impl.hpp"

namespace gmix {

template <int Dim> struct ExpectationMaximizationParameters {
  int n_components{0};
  int n_iterations{0};
  double early_stopping_threshold{0.0};
  bool warm_start{false};
};

template <int Dim> class ExpectationMaximizationPolicy {
public:
  using ParamType = ExpectationMaximizationParameters<Dim>;

  explicit ExpectationMaximizationPolicy(
      const ParamType &parameters) noexcept;

  explicit ExpectationMaximizationPolicy(ParamType &&parameters) noexcept;

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) const;

  void fit(std::vector<GaussianComponent<Dim>> &,
           const StaticRowsMatrix<Dim> &) const;

protected:
  ExpectationMaximizationPolicy() = default;

private:
  ParamType parameters_{};
};

template <int Dim>
ExpectationMaximizationPolicy<Dim>::ExpectationMaximizationPolicy(
    const ParamType &parameters) noexcept
    : parameters_{parameters} {}

template <int Dim>
ExpectationMaximizationPolicy<Dim>::ExpectationMaximizationPolicy(
    ParamType &&parameters) noexcept
    : parameters_{std::move(parameters)} {}

template <int Dim>
void ExpectationMaximizationPolicy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {

  const KMeansParameters<Dim> initialization_parameters = {
      parameters_.n_components, 1, 0.0, parameters_.warm_start};
  const auto initialization_policy =
      KMeansPolicy<Dim>{initialization_parameters};
  initialization_policy.fit(components, samples);
}

template <int Dim>
void ExpectationMaximizationPolicy<Dim>::fit(
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
    if (responsibilities.array().isNaN().any()) {
      initialize(components, samples);
      continue;
    }
    internal::estimate_parameters(samples, responsibilities, components);
    if (new_log_likelihood - current_log_likelihood <
        parameters_.early_stopping_threshold)
      return;
    else
      current_log_likelihood = new_log_likelihood;
  }
}

} // namespace gmix

#endif // !#ifndef GMIX_EXPECTATION_MAXIMIZATION_STRATEGY_HPP
