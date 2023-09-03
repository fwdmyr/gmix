#ifndef GMIX_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_IMPL_HPP
#define GMIX_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_IMPL_HPP

#include "kmeans_policy.hpp"
#include <cmath>
#include <cstdlib>
#include <eigen3/Eigen/src/Core/GlobalFunctions.h>
#include <eigen3/unsupported/Eigen/SpecialFunctions>

// TODO: There is a lot of optimization to be done here, also think about
// vectorizing more
// TODO: Implement early stopping based on variational lower bound

namespace gmix {

template <int Dim> struct VariationalBayesianInferenceParameters;
template <int Dim> struct RandomVariables;
template <int Dim> struct VariationalStatistics;

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

} // namespace internal

} // namespace gmix

#endif // !GMIX_VARIATIONAL_BAYESIAN_INFERENCE_STRATEGY_IMPL_HPP
