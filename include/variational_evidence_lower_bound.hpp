#ifndef GMIX_VARIATIONAL_EVIDENCE_LOWER_BOUND_HPP
#define GMIX_VARIATIONAL_EVIDENCE_LOWER_BOUND_HPP

#include "variational_bayesian_inference_policy_impl.hpp"

// TODO: There is a lot of optimization to be done here, also think about
// vectorizing more
// TODO: Implement early stopping based on variational lower bound

namespace gmix {

template <int Dim> struct VariationalBayesianInferenceParameters;
template <int Dim> struct RandomVariables;
template <int Dim> struct VariationalStatistics;

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

} // namespace gmix

#endif // !GMIX_VARIATIONAL_EVIDENCE_LOWER_BOUND_HPP
