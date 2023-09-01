#ifndef GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_HPP
#define GMSAM_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_HPP

#include "base_strategy.hpp"
#include "common.hpp"
#include "kmeans_strategy.hpp"
#include "statistics.hpp"
#include <cmath>
#include <cstdlib>
#include <eigen3/Eigen/src/Core/GlobalFunctions.h>
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#include <limits>
#include <random>
#include <variant>

// TODO: There is a lot of optimization to be done here, also think about
// vectorizing more
// TODO: Implement early stopping based on variational lower bound

namespace gmix {

template <int Dim> struct VariationalBayesianInferenceParameters;
template <int Dim> class VariationalBayesianInferenceStrategy;

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

namespace internal {

template <int Dim>
void evaluate_responsibilities(
    const RandomVariables<Dim> &random_variables,
    const StaticRowsMatrix<Dim> &samples,
    const VariationalBayesianInferenceParameters<Dim> &parameters,
    MatrixX &responsibilities) {
  const auto n_samples = samples.cols();
  for (size_t i = 0; i < parameters.n_components; ++i) {
    const auto e_logweight =
        Eigen::numext::digamma(random_variables.dirichlet_weight(i)) -
        Eigen::numext::digamma(random_variables.dirichlet_weight.sum());
    auto e_logdetinformation =
        Dim * std::log(2) + std::log(random_variables.wishart_information
                                         .block(0, i * Dim, Dim, Dim)
                                         .determinant());
    for (size_t d = 1; d <= Dim; ++d) {
      e_logdetinformation += Eigen::numext::digamma(
          0.5 * (random_variables.wishart_degrees_of_freedom(i) + 1 - d));
    }
    for (size_t j = 0; j < n_samples; ++j) {
      const auto distance =
          samples.col(j) - random_variables.normal_mean.col(i);
      const auto e_squaredmahalanobisdistance =
          static_cast<double>(Dim) /
              random_variables.normal_covariance_scaling(i) +
          random_variables.wishart_degrees_of_freedom(i) *
              distance.transpose() *
              random_variables.wishart_information.block(0, i * Dim, Dim, Dim) *
              distance;
      const auto logresponsibility =
          e_logweight + 0.5 * e_logdetinformation -
          0.5 * static_cast<double>(Dim) * std::log(2.0 * M_PI) -
          0.5 * e_squaredmahalanobisdistance;
      responsibilities(i, j) = std::exp(logresponsibility);
    }
  }
  // TODO: This can certainly be vectorized
  for (size_t j = 0; j < n_samples; ++j)
    responsibilities.col(j) =
        1.0 / responsibilities.col(j).sum() * responsibilities.col(j);
}

template <int Dim>
void compute_statistics(
    const MatrixX &responsibilities, const StaticRowsMatrix<Dim> &samples,
    const VariationalBayesianInferenceParameters<Dim> &parameters,
    VariationalStatistics<Dim> &statistics) {
  const auto n_samples = samples.cols();
  statistics.n_samples_responsible =
      static_cast<ColVectorX>(responsibilities.rowwise().sum());
  for (size_t i = 0; i < parameters.n_components; ++i) {
    statistics.mu.col(i) = 1.0 / statistics.n_samples_responsible(i) *
                           static_cast<ColVector<Dim>>(
                               (samples.transpose().array().colwise() *
                                responsibilities.row(i).transpose().array())
                                   .transpose()
                                   .rowwise()
                                   .sum());
    const auto centered_samples = static_cast<StaticRowsMatrix<Dim>>(
        samples.colwise() - statistics.mu.col(i));
    const auto weighted_centered_samples = static_cast<StaticRowsMatrix<Dim>>(
        (centered_samples.transpose().array().colwise() *
         responsibilities.row(i).transpose().array())
            .transpose());
    for (size_t j = 0; j < n_samples; ++j) {
      statistics.sigma.block(0, i * Dim, Dim, Dim) +=
          weighted_centered_samples.col(j) *
          centered_samples.transpose().row(j);
    }
    statistics.sigma.block(0, i * Dim, Dim, Dim) *=
        1.0 / statistics.n_samples_responsible(i);
  }
}

template <int Dim>
void update_random_variables(
    const VariationalStatistics<Dim> &statistics,
    const VariationalBayesianInferenceParameters<Dim> &parameters,
    RandomVariables<Dim> &random_variables) {

  random_variables.dirichlet_weight =
      ColVectorX::Constant(parameters.n_components, 1,
                           parameters.dirichlet_prior_weight) +
      statistics.n_samples_responsible;
  random_variables.normal_covariance_scaling =
      ColVectorX::Constant(parameters.n_components, 1,
                           parameters.normal_prior_covariance_scaling) +
      statistics.n_samples_responsible;
  for (size_t i = 0; i < parameters.n_components; ++i) {
    random_variables.normal_mean.col(i) =
        1.0 / random_variables.normal_covariance_scaling(i) *
        (parameters.normal_prior_covariance_scaling *
             parameters.normal_prior_mean +
         statistics.n_samples_responsible(i) * statistics.mu.col(i));
    random_variables.wishart_information.block(0, i * Dim, Dim, Dim) =
        (parameters.wishart_prior_information.inverse() +
         statistics.n_samples_responsible(i) *
             statistics.sigma.block(0, i * Dim, Dim, Dim) +
         parameters.normal_prior_covariance_scaling *
             statistics.n_samples_responsible(i) /
             (parameters.normal_prior_covariance_scaling +
              statistics.n_samples_responsible(i)) *
             (statistics.mu.col(i) - parameters.normal_prior_mean) *
             (statistics.mu.col(i) - parameters.normal_prior_mean).transpose())
            .inverse();
    random_variables.wishart_degrees_of_freedom(i) =
        parameters.wishart_prior_degrees_of_freedom +
        statistics.n_samples_responsible(i);
  }
}

double evaluate_sample_probabilitity_expectation();

double evaluate_hidden_probability_expectation();

double evaluate_weight_probability_expectation();

double evaluate_normal_probability_expectation();

double evaluate_hidden_posterior_expectation();

double evaluate_weight_posterior_expectation();

double evaluate_normal_posterior_expectation();

template <int Dim>
double evaluate_variational_lower_bound(
    const RandomVariables<Dim> &random_variables,
    const VariationalStatistics<Dim> &statistics,
    const MatrixX &responsibilities,
    const VariationalBayesianInferenceParameters<Dim> &parameters) {
  auto e_p_sample = 0.0;
  auto e_p_hidden = 0.0;
  auto e_p_weight =
      std::log(std::tgamma(random_variables.dirichlet_weight.sum()) /
               random_variables.dirichlet_weight
                   .template unaryExpr<double (*)(double)>(&std::exp)
                   .asDiagonal()
                   .toDenseMatrix()
                   .determinant());
  auto e_p_meaninformation = 0.0;
  auto e_q_hidden = 0.0;

  auto e_q_weight =
      std::log(std::tgamma(random_variables.dirichlet_weight.sum()) /
               random_variables.dirichlet_weight
                   .template unaryExpr<double (*)(double)>(&std::tgamma)
                   .asDiagonal()
                   .toDenseMatrix()
                   .determinant());
  auto e_q_meaninformation = 0.0;

  for (size_t k = 0; k < parameters.n_components; ++k) {
    const auto e_logweight =
        Eigen::numext::digamma(random_variables.dirichlet_weight(k)) -
        Eigen::numext::digamma(random_variables.dirichlet_weight.sum());
    auto e_logdetinformation =
        Dim * std::log(2) + std::log(random_variables.wishart_information
                                         .block(0, k * Dim, Dim, Dim)
                                         .determinant());
    for (size_t d = 1; d <= Dim; ++d) {
      e_logdetinformation += Eigen::numext::digamma(
          0.5 * (random_variables.wishart_degrees_of_freedom(k) + 1 - d));
    }

    e_p_sample +=
        0.5 * statistics.n_samples_responsible(k) * e_logdetinformation;
    e_p_sample += -0.5 * statistics.n_samples_responsible(k) * Dim /
                  random_variables.normal_covariance_scaling(k);
    e_p_sample +=
        -0.5 * statistics.n_samples_responsible(k) *
        (statistics.sigma.block(0, k * Dim, Dim, Dim) *
         random_variables.wishart_information.block(0, k * Dim, Dim, Dim))
            .trace();
    e_p_sample +=
        -0.5 * statistics.n_samples_responsible(k) *
        random_variables.wishart_degrees_of_freedom(k) *
        (statistics.mu.col(k) - random_variables.normal_mean.col(k))
            .transpose() *
        random_variables.wishart_information.block(0, k * Dim, Dim, Dim) *
        (statistics.mu.col(k) - random_variables.normal_mean.col(k));
    e_p_sample += -0.5 * statistics.n_samples_responsible(k) * Dim *
                  std::log(static_cast<double>(2.0 * M_PI));

    e_p_hidden += e_logweight * responsibilities.row(k).sum();

    e_p_weight += (parameters.dirichlet_prior_weight - 1.0) * e_logweight;

    // TODO: e_p_meaninformation

    e_q_hidden +=
        (responsibilities.row(k).transpose() *
         responsibilities.row(k).unaryExpr<double (*)(double)>(&std::log))
            .value();

    e_q_weight += (random_variables.dirichlet_weight(k) - 1.0) * e_logweight;

    // TODO: e_q_meaninformation
  }
  return evaluate_sample_probabilitity_expectation() +
         evaluate_hidden_probability_expectation() +
         evaluate_weight_probability_expectation() +
         evaluate_normal_probability_expectation() +
         evaluate_hidden_posterior_expectation() +
         evaluate_weight_posterior_expectation() +
         evaluate_normal_posterior_expectation();
}

} // namespace internal

template <int Dim>
class VariationalBayesianInferenceStrategy : public BaseStrategy<Dim> {
protected:
  using ParamType = VariationalBayesianInferenceParameters<Dim>;

  explicit VariationalBayesianInferenceStrategy(
      const VariationalBayesianInferenceParameters<Dim> &parameters) noexcept
      : parameters_(parameters) {}

  virtual void fit(std::vector<GaussianComponent<Dim>> &,
                   const StaticRowsMatrix<Dim> &) const override;

  void initialize(std::vector<GaussianComponent<Dim>> &,
                  const StaticRowsMatrix<Dim> &) override;

  VariationalBayesianInferenceParameters<Dim> parameters_{};
};

template <int Dim>
void VariationalBayesianInferenceStrategy<Dim>::initialize(
    std::vector<GaussianComponent<Dim>> &components,
    const StaticRowsMatrix<Dim> &samples) {
  const KMeansParameters<Dim> initialization_parameters = {
      parameters_.n_components, 1, 0.0, parameters_.warm_start};
  const auto initialization_strategy =
      KMeansStrategy<Dim>{initialization_parameters};
  initialization_strategy.fit(components, samples);
  parameters_.warm_start = true;
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
