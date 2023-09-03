#ifndef GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_HPP
#define GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_HPP

#include "variational_bayesian_inference_strategy_impl.hpp"

// TODO: There is a lot of optimization to be done here, also think about
// vectorizing more
// TODO: Implement early stopping based on variational lower bound

namespace gmix {

template <int Dim> struct VariationalBayesianInferenceParameters {
  int n_components{0};
  int n_iterations{0};
  double early_stopping_threshold{0.0};
  bool warm_start{false};
  double dirichlet_prior_weight{};
  ColVector<Dim> normal_prior_mean{};
  double normal_prior_covariance_scaling{};
  double wishart_prior_degrees_of_freedom{};
  Matrix<Dim, Dim> wishart_prior_information{};
};

template <int Dim> struct RandomVariables {

  explicit RandomVariables(const VariationalBayesianInferenceParameters<Dim> &);

  ColVectorX dirichlet_weight{};
  StaticRowsMatrix<Dim> normal_mean{};
  ColVectorX normal_covariance_scaling{};
  StaticRowsMatrix<Dim> wishart_information{};
  ColVectorX wishart_degrees_of_freedom{};
};

template <int Dim>
RandomVariables<Dim>::RandomVariables(
    const VariationalBayesianInferenceParameters<Dim> &parameters)
    : dirichlet_weight{parameters.dirichlet_prior_weight *
                       ColVectorX::Ones(parameters.n_components, 1)},
      normal_mean{
          parameters.normal_prior_mean.replicate(1, parameters.n_components)},
      normal_covariance_scaling{parameters.normal_prior_covariance_scaling *
                                ColVectorX::Ones(parameters.n_components, 1)},
      wishart_information{parameters.wishart_prior_information.replicate(
          1, parameters.n_components)},
      wishart_degrees_of_freedom{parameters.wishart_prior_degrees_of_freedom *
                                 ColVectorX::Ones(parameters.n_components, 1)} {
}

template <int Dim> struct VariationalStatistics {

  explicit VariationalStatistics(
      const VariationalBayesianInferenceParameters<Dim> &);

  ColVectorX n_samples_responsible{};
  StaticRowsMatrix<Dim> mu{};
  StaticRowsMatrix<Dim> sigma{};
};

template <int Dim>
VariationalStatistics<Dim>::VariationalStatistics(
    const VariationalBayesianInferenceParameters<Dim> &parameters)
    : n_samples_responsible{ColVectorX::Zero(parameters.n_components, 1)},
      mu{StaticRowsMatrix<Dim>::Zero(Dim, parameters.n_components)},
      sigma{StaticRowsMatrix<Dim>::Zero(Dim, Dim * parameters.n_components)} {}

template <int Dim> class VariationalBayesianInferenceStrategy {
public:
  using ParamType = VariationalBayesianInferenceParameters<Dim>;

  explicit VariationalBayesianInferenceStrategy(ParamType &&parameters) noexcept
      : parameters_(std::move(parameters)) {}

  explicit VariationalBayesianInferenceStrategy(
      const ParamType &parameters) noexcept
      : parameters_(parameters) {}

  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const;

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) const;

protected:
  VariationalBayesianInferenceStrategy() = default;

private:
  ParamType parameters_{};
};

template <int Dim>
void VariationalBayesianInferenceStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {
  const KMeansParameters<Dim> initialization_parameters = {
      parameters_.n_components, 1, 0.0, parameters_.warm_start};
  const auto initialization_strategy =
      KMeansStrategy<Dim>{initialization_parameters};
  initialization_strategy.fit(components, samples);
}

template <int Dim>
void VariationalBayesianInferenceStrategy<Dim>::fit(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) const {

  const auto n_samples = samples.cols();
  const auto n_components = parameters_.n_components;

  auto random_variables = RandomVariables<Dim>{parameters_};
  auto statistics = VariationalStatistics<Dim>{parameters_};

  if (!parameters_.warm_start || components.size() != n_components)
    initialize(components, samples);

  for (size_t i = 0; i < n_components; ++i) {
    const auto &component = components[i];
    random_variables.dirichlet_weight(i) = component.get_weight();
    random_variables.normal_mean.col(i) = component.get_mean();
    random_variables.wishart_information.block(0, i * Dim, Dim, Dim) =
        component.get_covariance().inverse();
  }

  auto responsibilities =
      static_cast<MatrixX>(MatrixX::Zero(n_components, n_samples));

  for (size_t i = 0; i < parameters_.n_iterations; ++i) {

    internal::evaluate_responsibilities(random_variables, samples, parameters_,
                                        responsibilities);

    if (responsibilities.array().isNaN().any()) {
      initialize(components, samples);
      continue;
    }

    internal::compute_statistics(responsibilities, samples, parameters_,
                                 statistics);

    internal::update_random_variables(statistics, parameters_,
                                      random_variables);
  }

  for (size_t i = 0; i < n_components; ++i) {
    auto &component = components[i];
    component.set_weight(random_variables.dirichlet_weight(i) /
                         random_variables.dirichlet_weight.sum());
    component.set_mean(random_variables.normal_mean.col(i));
    component.set_covariance(
        (random_variables.wishart_degrees_of_freedom(i) *
         random_variables.wishart_information.block(0, i * Dim, Dim, Dim))
            .inverse());
  }
}

} // namespace gmix

#endif // !GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_HPP
